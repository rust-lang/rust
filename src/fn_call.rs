use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, Align, LayoutOf, Size};
use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc::mir;
use rustc_data_structures::indexed_vec::Idx;
use rustc_target::spec::abi::Abi;
use syntax::attr;
use syntax::codemap::Span;

use std::mem;

use super::*;

use tls::MemoryExt;

use super::memory::MemoryKind;

fn write_discriminant_value<'a, 'mir, 'tcx: 'a + 'mir>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, super::Evaluator<'tcx>>,
        dest_ty: Ty<'tcx>,
        dest: Place,
        variant_index: usize,
    ) -> EvalResult<'tcx> {
        let layout = ecx.layout_of(dest_ty)?;

        match layout.variants {
            layout::Variants::Single { index } => {
                if index != variant_index {
                    // If the layout of an enum is `Single`, all
                    // other variants are necessarily uninhabited.
                    assert_eq!(layout.for_variant(&ecx, variant_index).abi,
                               layout::Abi::Uninhabited);
                }
            }
            layout::Variants::Tagged { .. } => {
                let discr_val = dest_ty.ty_adt_def().unwrap()
                    .discriminant_for_variant(*ecx.tcx, variant_index)
                    .val;

                let (discr_dest, discr) = ecx.place_field(dest, mir::Field::new(0), layout)?;
                ecx.write_scalar(discr_dest, Scalar::from_u128(discr_val), discr.ty)?;
            }
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                if variant_index != dataful_variant {
                    let (niche_dest, niche) =
                        ecx.place_field(dest, mir::Field::new(0), layout)?;
                    let niche_value = ((variant_index - niche_variants.start()) as u128)
                        .wrapping_add(niche_start);
                    ecx.write_scalar(niche_dest, Scalar::from_u128(niche_value), niche.ty)?;
                }
            }
        }

        Ok(())
    }

pub trait EvalContextExt<'tcx> {
    fn call_c_abi(
        &mut self,
        def_id: DefId,
        args: &[ValTy<'tcx>],
        dest: Place,
        dest_ty: Ty<'tcx>,
        dest_block: mir::BasicBlock,
    ) -> EvalResult<'tcx>;

    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>>;

    fn call_missing_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Place, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx>;

    fn eval_fn_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Place, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool>;

    fn write_null(&mut self, dest: Place, dest_ty: Ty<'tcx>) -> EvalResult<'tcx>;
}

impl<'a, 'mir, 'tcx: 'mir + 'a> EvalContextExt<'tcx> for EvalContext<'a, 'mir, 'tcx, super::Evaluator<'tcx>> {
    fn eval_fn_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Place, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        trace!("eval_fn_call: {:#?}, {:#?}", instance, destination);

        let def_id = instance.def_id();
        let item_path = self.tcx.absolute_item_path_str(def_id);
        if item_path.starts_with("std::") {
            //println!("{}", item_path);
        }
        match &*item_path {
            "std::sys::unix::thread::guard::init" | "std::sys::unix::thread::guard::current" => {
                // Return None, as it doesn't make sense to return Some, because miri detects stack overflow itself.
                let ret_ty = sig.output();
                match ret_ty.sty {
                    ty::TyAdt(ref adt_def, _) => {
                        assert!(adt_def.is_enum(), "Unexpected return type for {}", item_path);
                        let none_variant_index = adt_def.variants.iter().position(|def| {
                            def.name.as_str() == "None"
                        }).expect("No None variant");
                        let (return_place, return_to_block) = destination.unwrap();
                        write_discriminant_value(self, ret_ty, return_place, none_variant_index)?;
                        self.goto_block(return_to_block);
                        return Ok(true);
                    }
                    _ => panic!("Unexpected return type for {}", item_path)
                }
            }
            "std::sys::unix::fast_thread_local::register_dtor" => {
                // TODO: register the dtor
                let (_return_place, return_to_block) = destination.unwrap();
                self.goto_block(return_to_block);
                return Ok(true);
            }
            _ => {}
        }

        let mir = match self.load_mir(instance.def) {
            Ok(mir) => mir,
            Err(EvalError { kind: EvalErrorKind::NoMirFor(path), .. }) => {
                self.call_missing_fn(
                    instance,
                    destination,
                    args,
                    sig,
                    path,
                )?;
                return Ok(true);
            }
            Err(other) => return Err(other),
        };

        let (return_place, return_to_block) = match destination {
            Some((place, block)) => (place, StackPopCleanup::Goto(block)),
            None => (Place::undef(), StackPopCleanup::None),
        };

        self.push_stack_frame(
            instance,
            span,
            mir,
            return_place,
            return_to_block,
        )?;

        Ok(false)
    }

    fn call_c_abi(
        &mut self,
        def_id: DefId,
        args: &[ValTy<'tcx>],
        dest: Place,
        dest_ty: Ty<'tcx>,
        dest_block: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let attrs = self.tcx.get_attrs(def_id);
        let link_name = match attr::first_attr_value_str_by_name(&attrs, "link_name") {
            Some(name) => name.as_str(),
            None => self.tcx.item_name(def_id).as_str(),
        };

        match &link_name[..] {
            "malloc" => {
                let size = self.value_to_scalar(args[0])?.to_u64()?;
                if size == 0 {
                    self.write_null(dest, dest_ty)?;
                } else {
                    let align = self.tcx.data_layout.pointer_align;
                    let ptr = self.memory.allocate(Size::from_bytes(size), align, Some(MemoryKind::C.into()))?;
                    self.write_scalar(dest, Scalar::Ptr(ptr), dest_ty)?;
                }
            }

            "free" => {
                let ptr = self.into_ptr(args[0].value)?;
                if !ptr.is_null()? {
                    self.memory.deallocate(
                        ptr.to_ptr()?,
                        None,
                        MemoryKind::C.into(),
                    )?;
                }
            }

            "syscall" => {
                // TODO: read `syscall` ids like `sysconf` ids and
                // figure out some way to actually process some of them
                //
                // libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)
                // is called if a `HashMap` is created the regular way.
                match self.value_to_scalar(args[0])?.to_u64()? {
                    318 | 511 => {
                        return err!(Unimplemented(
                            "miri does not support random number generators".to_owned(),
                        ))
                    }
                    id => {
                        return err!(Unimplemented(
                            format!("miri does not support syscall id {}", id),
                        ))
                    }
                }
            }

            "dlsym" => {
                let _handle = self.into_ptr(args[0].value)?;
                let symbol = self.into_ptr(args[1].value)?.to_ptr()?;
                let symbol_name = self.memory.read_c_str(symbol)?;
                let err = format!("bad c unicode symbol: {:?}", symbol_name);
                let symbol_name = ::std::str::from_utf8(symbol_name).unwrap_or(&err);
                return err!(Unimplemented(format!(
                    "miri does not support dynamically loading libraries (requested symbol: {})",
                    symbol_name
                )));
            }

            "__rust_maybe_catch_panic" => {
                // fn __rust_maybe_catch_panic(f: fn(*mut u8), data: *mut u8, data_ptr: *mut usize, vtable_ptr: *mut usize) -> u32
                // We abort on panic, so not much is going on here, but we still have to call the closure
                let u8_ptr_ty = self.tcx.mk_mut_ptr(self.tcx.types.u8);
                let f = self.into_ptr(args[0].value)?.to_ptr()?;
                let data = self.into_ptr(args[1].value)?;
                let f_instance = self.memory.get_fn(f)?;
                self.write_null(dest, dest_ty)?;

                // Now we make a function call.  TODO: Consider making this re-usable?  EvalContext::step does sth. similar for the TLS dtors,
                // and of course eval_main.
                let mir = self.load_mir(f_instance.def)?;
                self.push_stack_frame(
                    f_instance,
                    mir.span,
                    mir,
                    Place::undef(),
                    StackPopCleanup::Goto(dest_block),
                )?;
                let mut args = self.frame().mir.args_iter();

                let arg_local = args.next().ok_or(
                    EvalErrorKind::AbiViolation(
                        "Argument to __rust_maybe_catch_panic does not take enough arguments."
                            .to_owned(),
                    ),
                )?;
                let arg_dest = self.eval_place(&mir::Place::Local(arg_local))?;
                self.write_ptr(arg_dest, data, u8_ptr_ty)?;

                assert!(args.next().is_none(), "__rust_maybe_catch_panic argument has more arguments than expected");

                // We ourselves return 0
                self.write_null(dest, dest_ty)?;

                // Don't fall through
                return Ok(());
            }

            "__rust_start_panic" => {
                return err!(Panic);
            }

            "memcmp" => {
                let left = self.into_ptr(args[0].value)?;
                let right = self.into_ptr(args[1].value)?;
                let n = Size::from_bytes(self.value_to_scalar(args[2])?.to_u64()?);

                let result = {
                    let left_bytes = self.memory.read_bytes(left, n)?;
                    let right_bytes = self.memory.read_bytes(right, n)?;

                    use std::cmp::Ordering::*;
                    match left_bytes.cmp(right_bytes) {
                        Less => -1i32,
                        Equal => 0,
                        Greater => 1,
                    }
                };

                self.write_scalar(
                    dest,
                    Scalar::from_i32(result),
                    dest_ty,
                )?;
            }

            "memrchr" => {
                let ptr = self.into_ptr(args[0].value)?;
                let val = self.value_to_scalar(args[1])?.to_u64()? as u8;
                let num = self.value_to_scalar(args[2])?.to_u64()?;
                if let Some(idx) = self.memory.read_bytes(ptr, Size::from_bytes(num))?.iter().rev().position(
                    |&c| c == val,
                )
                {
                    let new_ptr = ptr.ptr_offset(Size::from_bytes(num - idx as u64 - 1), &self)?;
                    self.write_ptr(dest, new_ptr, dest_ty)?;
                } else {
                    self.write_null(dest, dest_ty)?;
                }
            }

            "memchr" => {
                let ptr = self.into_ptr(args[0].value)?;
                let val = self.value_to_scalar(args[1])?.to_u64()? as u8;
                let num = self.value_to_scalar(args[2])?.to_u64()?;
                if let Some(idx) = self.memory.read_bytes(ptr, Size::from_bytes(num))?.iter().position(
                    |&c| c == val,
                )
                {
                    let new_ptr = ptr.ptr_offset(Size::from_bytes(idx as u64), &self)?;
                    self.write_ptr(dest, new_ptr, dest_ty)?;
                } else {
                    self.write_null(dest, dest_ty)?;
                }
            }

            "getenv" => {
                let result = {
                    let name_ptr = self.into_ptr(args[0].value)?.to_ptr()?;
                    let name = self.memory.read_c_str(name_ptr)?;
                    match self.machine.env_vars.get(name) {
                        Some(&var) => Scalar::Ptr(var),
                        None => Scalar::null(),
                    }
                };
                self.write_scalar(dest, result, dest_ty)?;
            }

            "unsetenv" => {
                let mut success = None;
                {
                    let name_ptr = self.into_ptr(args[0].value)?;
                    if !name_ptr.is_null()? {
                        let name = self.memory.read_c_str(name_ptr.to_ptr()?)?;
                        if !name.is_empty() && !name.contains(&b'=') {
                            success = Some(self.machine.env_vars.remove(name));
                        }
                    }
                }
                if let Some(old) = success {
                    if let Some(var) = old {
                        self.memory.deallocate(var, None, MemoryKind::Env.into())?;
                    }
                    self.write_null(dest, dest_ty)?;
                } else {
                    self.write_scalar(dest, Scalar::from_i128(-1), dest_ty)?;
                }
            }

            "setenv" => {
                let mut new = None;
                {
                    let name_ptr = self.into_ptr(args[0].value)?;
                    let value_ptr = self.into_ptr(args[1].value)?.to_ptr()?;
                    let value = self.memory.read_c_str(value_ptr)?;
                    if !name_ptr.is_null()? {
                        let name = self.memory.read_c_str(name_ptr.to_ptr()?)?;
                        if !name.is_empty() && !name.contains(&b'=') {
                            new = Some((name.to_owned(), value.to_owned()));
                        }
                    }
                }
                if let Some((name, value)) = new {
                    // +1 for the null terminator
                    let value_copy = self.memory.allocate(
                        Size::from_bytes((value.len() + 1) as u64),
                        Align::from_bytes(1, 1).unwrap(),
                        Some(MemoryKind::Env.into()),
                    )?;
                    self.memory.write_bytes(value_copy.into(), &value)?;
                    let trailing_zero_ptr = value_copy.offset(Size::from_bytes(value.len() as u64), &self)?.into();
                    self.memory.write_bytes(trailing_zero_ptr, &[0])?;
                    if let Some(var) = self.machine.env_vars.insert(
                        name.to_owned(),
                        value_copy,
                    )
                    {
                        self.memory.deallocate(var, None, MemoryKind::Env.into())?;
                    }
                    self.write_null(dest, dest_ty)?;
                } else {
                    self.write_scalar(dest, Scalar::from_i128(-1), dest_ty)?;
                }
            }

            "write" => {
                let fd = self.value_to_scalar(args[0])?.to_u64()?;
                let buf = self.into_ptr(args[1].value)?;
                let n = self.value_to_scalar(args[2])?.to_u64()?;
                trace!("Called write({:?}, {:?}, {:?})", fd, buf, n);
                let result = if fd == 1 || fd == 2 {
                    // stdout/stderr
                    use std::io::{self, Write};

                    let buf_cont = self.memory.read_bytes(buf, Size::from_bytes(n))?;
                    let res = if fd == 1 {
                        io::stdout().write(buf_cont)
                    } else {
                        io::stderr().write(buf_cont)
                    };
                    match res {
                        Ok(n) => n as i64,
                        Err(_) => -1,
                    }
                } else {
                    warn!("Ignored output to FD {}", fd);
                    n as i64 // pretend it all went well
                }; // now result is the value we return back to the program
                let ptr_size = self.memory.pointer_size();
                self.write_scalar(
                    dest,
                    Scalar::from_isize(result, ptr_size),
                    dest_ty,
                )?;
            }

            "strlen" => {
                let ptr = self.into_ptr(args[0].value)?.to_ptr()?;
                let n = self.memory.read_c_str(ptr)?.len();
                let ptr_size = self.memory.pointer_size();
                self.write_scalar(dest, Scalar::from_usize(n as u64, ptr_size), dest_ty)?;
            }

            // Some things needed for sys::thread initialization to go through
            "signal" | "sigaction" | "sigaltstack" => {
                self.write_scalar(dest, Scalar::null(), dest_ty)?;
            }

            "sysconf" => {
                let name = self.value_to_scalar(args[0])?.to_u64()?;

                trace!("sysconf() called with name {}", name);
                // cache the sysconf integers via miri's global cache
                let paths = &[
                    (&["libc", "_SC_PAGESIZE"], Scalar::from_i128(4096)),
                    (&["libc", "_SC_GETPW_R_SIZE_MAX"], Scalar::from_i128(-1)),
                ];
                let mut result = None;
                for &(path, path_value) in paths {
                    if let Ok(instance) = self.resolve_path(path) {
                        let cid = GlobalId {
                            instance,
                            promoted: None,
                        };
                        let const_val = self.const_eval(cid)?;
                        let value = const_val.unwrap_usize(self.tcx.tcx);
                        if value == name {
                            result = Some(path_value);
                            break;
                        }
                    }
                }
                if let Some(result) = result {
                    self.write_scalar(dest, result, dest_ty)?;
                } else {
                    return err!(Unimplemented(
                        format!("Unimplemented sysconf name: {}", name),
                    ));
                }
            }

            // Hook pthread calls that go to the thread-local storage memory subsystem
            "pthread_key_create" => {
                let key_ptr = self.into_ptr(args[0].value)?;
                let key_align = self.layout_of(args[0].ty)?.align;

                // Extract the function type out of the signature (that seems easier than constructing it ourselves...)
                let dtor = match self.into_ptr(args[1].value)? {
                    Scalar::Ptr(dtor_ptr) => Some(self.memory.get_fn(dtor_ptr)?),
                    Scalar::Bits { defined: 0, .. } => return err!(ReadUndefBytes),
                    Scalar::Bits { bits: 0, .. } => None,
                    Scalar::Bits { .. } => return err!(ReadBytesAsPointer),
                };

                // Figure out how large a pthread TLS key actually is. This is libc::pthread_key_t.
                let key_type = args[0].ty.builtin_deref(true)
                                   .ok_or(EvalErrorKind::AbiViolation("Wrong signature used for pthread_key_create: First argument must be a raw pointer.".to_owned()))?.ty;
                let key_size = self.layout_of(key_type)?.size;

                // Create key and write it into the memory where key_ptr wants it
                let key = self.memory.create_tls_key(dtor) as u128;
                if key_size.bits() < 128 && key >= (1u128 << key_size.bits() as u128) {
                    return err!(OutOfTls);
                }
                self.memory.write_scalar(
                    key_ptr,
                    key_align,
                    Scalar::from_u128(key),
                    key_size,
                    false,
                )?;

                // Return success (0)
                self.write_null(dest, dest_ty)?;
            }
            "pthread_key_delete" => {
                let key = self.value_to_scalar(args[0])?.to_bytes()?;
                self.memory.delete_tls_key(key)?;
                // Return success (0)
                self.write_null(dest, dest_ty)?;
            }
            "pthread_getspecific" => {
                let key = self.value_to_scalar(args[0])?.to_bytes()?;
                let ptr = self.memory.load_tls(key)?;
                self.write_ptr(dest, ptr, dest_ty)?;
            }
            "pthread_setspecific" => {
                let key = self.value_to_scalar(args[0])?.to_bytes()?;
                let new_ptr = self.into_ptr(args[1].value)?;
                self.memory.store_tls(key, new_ptr)?;

                // Return success (0)
                self.write_null(dest, dest_ty)?;
            }

            "_tlv_atexit" => {
                return err!(Unimplemented("can't interpret with full mir for osx target".to_owned()));
            },

            // Stub out all the other pthread calls to just return 0
            link_name if link_name.starts_with("pthread_") => {
                info!("ignoring C ABI call: {}", link_name);
                self.write_null(dest, dest_ty)?;
            }

            "mmap" => {
                // This is a horrible hack, but well... the guard page mechanism calls mmap and expects a particular return value, so we give it that value
                let addr = self.into_ptr(args[0].value)?;
                self.write_ptr(dest, addr, dest_ty)?;
            }

            _ => {
                return err!(Unimplemented(
                    format!("can't call C ABI function: {}", link_name),
                ));
            }
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        self.dump_local(dest);
        self.goto_block(dest_block);
        Ok(())
    }

    /// Get an instance for a path.
    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        self.tcx
            .crates()
            .iter()
            .find(|&&krate| self.tcx.original_crate_name(krate) == path[0])
            .and_then(|krate| {
                let krate = DefId {
                    krate: *krate,
                    index: CRATE_DEF_INDEX,
                };
                let mut items = self.tcx.item_children(krate);
                let mut path_it = path.iter().skip(1).peekable();

                while let Some(segment) = path_it.next() {
                    for item in mem::replace(&mut items, Default::default()).iter() {
                        if item.ident.name == *segment {
                            if path_it.peek().is_none() {
                                return Some(ty::Instance::mono(self.tcx.tcx, item.def.def_id()));
                            }

                            items = self.tcx.item_children(item.def.def_id());
                            break;
                        }
                    }
                }
                None
            })
            .ok_or_else(|| {
                let path = path.iter().map(|&s| s.to_owned()).collect();
                EvalErrorKind::PathNotFound(path).into()
            })
    }

    fn call_missing_fn(
        &mut self,
        instance: ty::Instance<'tcx>,
        destination: Option<(Place, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx> {
        // In some cases in non-MIR libstd-mode, not having a destination is legit.  Handle these early.
        match &path[..] {
            "std::panicking::rust_panic_with_hook" |
            "core::panicking::panic_fmt::::panic_impl" |
            "std::rt::begin_panic_fmt" => return err!(Panic),
            _ => {}
        }

        let dest_ty = sig.output();
        let (dest, dest_block) = destination.ok_or_else(
            || EvalErrorKind::NoMirFor(path.clone()),
        )?;

        if sig.abi == Abi::C {
            // An external C function
            // TODO: That functions actually has a similar preamble to what follows here.  May make sense to
            // unify these two mechanisms for "hooking into missing functions".
            self.call_c_abi(
                instance.def_id(),
                args,
                dest,
                dest_ty,
                dest_block,
            )?;
            return Ok(());
        }

        match &path[..] {
            // Allocators are magic.  They have no MIR, even when the rest of libstd does.
            "alloc::alloc::::__rust_alloc" => {
                let size = self.value_to_scalar(args[0])?.to_u64()?;
                let align = self.value_to_scalar(args[1])?.to_u64()?;
                if size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = self.memory.allocate(Size::from_bytes(size),
                                               Align::from_bytes(align, align).unwrap(),
                                               Some(MemoryKind::Rust.into()))?;
                self.write_scalar(dest, Scalar::Ptr(ptr), dest_ty)?;
            }
            "alloc::alloc::::__rust_alloc_zeroed" => {
                let size = self.value_to_scalar(args[0])?.to_u64()?;
                let align = self.value_to_scalar(args[1])?.to_u64()?;
                if size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let ptr = self.memory.allocate(Size::from_bytes(size),
                                               Align::from_bytes(align, align).unwrap(),
                                               Some(MemoryKind::Rust.into()))?;
                self.memory.write_repeat(ptr.into(), 0, Size::from_bytes(size))?;
                self.write_scalar(dest, Scalar::Ptr(ptr), dest_ty)?;
            }
            "alloc::alloc::::__rust_dealloc" => {
                let ptr = self.into_ptr(args[0].value)?.to_ptr()?;
                let old_size = self.value_to_scalar(args[1])?.to_u64()?;
                let align = self.value_to_scalar(args[2])?.to_u64()?;
                if old_size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                self.memory.deallocate(
                    ptr,
                    Some((Size::from_bytes(old_size), Align::from_bytes(align, align).unwrap())),
                    MemoryKind::Rust.into(),
                )?;
            }
            "alloc::alloc::::__rust_realloc" => {
                let ptr = self.into_ptr(args[0].value)?.to_ptr()?;
                let old_size = self.value_to_scalar(args[1])?.to_u64()?;
                let align = self.value_to_scalar(args[2])?.to_u64()?;
                let new_size = self.value_to_scalar(args[3])?.to_u64()?;
                if old_size == 0 || new_size == 0 {
                    return err!(HeapAllocZeroBytes);
                }
                if !align.is_power_of_two() {
                    return err!(HeapAllocNonPowerOfTwoAlignment(align));
                }
                let new_ptr = self.memory.reallocate(
                    ptr,
                    Size::from_bytes(old_size),
                    Align::from_bytes(align, align).unwrap(),
                    Size::from_bytes(new_size),
                    Align::from_bytes(align, align).unwrap(),
                    MemoryKind::Rust.into(),
                )?;
                self.write_scalar(dest, Scalar::Ptr(new_ptr), dest_ty)?;
            }

            // A Rust function is missing, which means we are running with MIR missing for libstd (or other dependencies).
            // Still, we can make many things mostly work by "emulating" or ignoring some functions.
            "std::io::_print" => {
                warn!(
                    "Ignoring output.  To run programs that print, make sure you have a libstd with full MIR."
                );
            }
            "std::thread::Builder::new" => {
                return err!(Unimplemented("miri does not support threading".to_owned()))
            }
            "std::env::args" => {
                return err!(Unimplemented(
                    "miri does not support program arguments".to_owned(),
                ))
            }
            "std::panicking::panicking" |
            "std::rt::panicking" => {
                // we abort on panic -> `std::rt::panicking` always returns false
                let bool = self.tcx.types.bool;
                self.write_scalar(dest, Scalar::from_bool(false), bool)?;
            }
            "std::sys::imp::c::::AddVectoredExceptionHandler" |
            "std::sys::imp::c::::SetThreadStackGuarantee" => {
                let usize = self.tcx.types.usize;
                // any non zero value works for the stdlib. This is just used for stackoverflows anyway
                self.write_scalar(dest, Scalar::from_u128(1), usize)?;
            },
            _ => return err!(NoMirFor(path)),
        }

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        self.dump_local(dest);
        self.goto_block(dest_block);
        return Ok(());
    }

    fn write_null(&mut self, dest: Place, dest_ty: Ty<'tcx>) -> EvalResult<'tcx> {
        self.write_scalar(dest, Scalar::null(), dest_ty)
    }
}
