//! Interpret intrinsics, lang items and `extern "C"` wellknown functions which their implementation
//! is not available.
//!
use std::cmp::{self, Ordering};

use chalk_ir::TyKind;
use hir_def::{
    builtin_type::{BuiltinInt, BuiltinUint},
    lang_item::LangItemTarget,
    resolver::HasResolver,
};
use hir_expand::name::Name;
use intern::{sym, Symbol};
use stdx::never;

use crate::{
    display::DisplayTarget,
    error_lifetime,
    mir::eval::{
        pad16, Address, AdtId, Arc, BuiltinType, Evaluator, FunctionId, HasModule, HirDisplay,
        InternedClosure, Interner, Interval, IntervalAndTy, IntervalOrOwned, ItemContainerId,
        LangItem, Layout, Locals, Lookup, MirEvalError, MirSpan, Mutability, Result, Substitution,
        Ty, TyBuilder, TyExt,
    },
    DropGlue,
};

mod simd;

macro_rules! from_bytes {
    ($ty:tt, $value:expr) => {
        ($ty::from_le_bytes(match ($value).try_into() {
            Ok(it) => it,
            #[allow(unreachable_patterns)]
            Err(_) => return Err(MirEvalError::InternalError("mismatched size".into())),
        }))
    };
}

macro_rules! not_supported {
    ($it: expr) => {
        return Err(MirEvalError::NotSupported(format!($it)))
    };
}

impl Evaluator<'_> {
    pub(super) fn detect_and_exec_special_function(
        &mut self,
        def: FunctionId,
        args: &[IntervalAndTy],
        generic_args: &Substitution,
        locals: &Locals,
        destination: Interval,
        span: MirSpan,
    ) -> Result<bool> {
        if self.not_special_fn_cache.borrow().contains(&def) {
            return Ok(false);
        }

        let function_data = self.db.function_data(def);
        let attrs = self.db.attrs(def.into());
        let is_intrinsic = attrs.by_key(&sym::rustc_intrinsic).exists()
            // Keep this around for a bit until extern "rustc-intrinsic" abis are no longer used
            || (match &function_data.abi {
                Some(abi) => *abi == sym::rust_dash_intrinsic,
                None => match def.lookup(self.db.upcast()).container {
                    hir_def::ItemContainerId::ExternBlockId(block) => {
                        let id = block.lookup(self.db.upcast()).id;
                        id.item_tree(self.db.upcast())[id.value].abi.as_ref()
                            == Some(&sym::rust_dash_intrinsic)
                    }
                    _ => false,
                },
            });

        if is_intrinsic {
            return self.exec_intrinsic(
                function_data.name.as_str(),
                args,
                generic_args,
                destination,
                locals,
                span,
                !function_data.has_body()
                    || attrs.by_key(&sym::rustc_intrinsic_must_be_overridden).exists(),
            );
        }
        let is_extern_c = match def.lookup(self.db.upcast()).container {
            hir_def::ItemContainerId::ExternBlockId(block) => {
                let id = block.lookup(self.db.upcast()).id;
                id.item_tree(self.db.upcast())[id.value].abi.as_ref() == Some(&sym::C)
            }
            _ => false,
        };
        if is_extern_c {
            return self
                .exec_extern_c(
                    function_data.name.as_str(),
                    args,
                    generic_args,
                    destination,
                    locals,
                    span,
                )
                .map(|()| true);
        }

        let alloc_fn =
            attrs.iter().filter_map(|it| it.path().as_ident()).map(|it| it.symbol()).find(|it| {
                [
                    &sym::rustc_allocator,
                    &sym::rustc_deallocator,
                    &sym::rustc_reallocator,
                    &sym::rustc_allocator_zeroed,
                ]
                .contains(it)
            });
        if let Some(alloc_fn) = alloc_fn {
            self.exec_alloc_fn(alloc_fn, args, destination)?;
            return Ok(true);
        }
        if let Some(it) = self.detect_lang_function(def) {
            let result = self.exec_lang_item(it, generic_args, args, locals, span)?;
            destination.write_from_bytes(self, &result)?;
            return Ok(true);
        }
        if let ItemContainerId::TraitId(t) = def.lookup(self.db.upcast()).container {
            if self.db.lang_attr(t.into()) == Some(LangItem::Clone) {
                let [self_ty] = generic_args.as_slice(Interner) else {
                    not_supported!("wrong generic arg count for clone");
                };
                let Some(self_ty) = self_ty.ty(Interner) else {
                    not_supported!("wrong generic arg kind for clone");
                };
                // Clone has special impls for tuples and function pointers
                if matches!(
                    self_ty.kind(Interner),
                    TyKind::Function(_) | TyKind::Tuple(..) | TyKind::Closure(..)
                ) {
                    self.exec_clone(def, args, self_ty.clone(), locals, destination, span)?;
                    return Ok(true);
                }
                // Return early to prevent caching clone as non special fn.
                return Ok(false);
            }
        }
        self.not_special_fn_cache.borrow_mut().insert(def);
        Ok(false)
    }

    pub(super) fn detect_and_redirect_special_function(
        &mut self,
        def: FunctionId,
    ) -> Result<Option<FunctionId>> {
        // `PanicFmt` is redirected to `ConstPanicFmt`
        if let Some(LangItem::PanicFmt) = self.db.lang_attr(def.into()) {
            let resolver =
                self.db.crate_def_map(self.crate_id).crate_root().resolver(self.db.upcast());

            let Some(hir_def::lang_item::LangItemTarget::Function(const_panic_fmt)) =
                self.db.lang_item(resolver.krate(), LangItem::ConstPanicFmt)
            else {
                not_supported!("const_panic_fmt lang item not found or not a function");
            };
            return Ok(Some(const_panic_fmt));
        }
        Ok(None)
    }

    /// Clone has special impls for tuples and function pointers
    fn exec_clone(
        &mut self,
        def: FunctionId,
        args: &[IntervalAndTy],
        self_ty: Ty,
        locals: &Locals,
        destination: Interval,
        span: MirSpan,
    ) -> Result<()> {
        match self_ty.kind(Interner) {
            TyKind::Function(_) => {
                let [arg] = args else {
                    not_supported!("wrong arg count for clone");
                };
                let addr = Address::from_bytes(arg.get(self)?)?;
                return destination
                    .write_from_interval(self, Interval { addr, size: destination.size });
            }
            TyKind::Closure(id, subst) => {
                let [arg] = args else {
                    not_supported!("wrong arg count for clone");
                };
                let addr = Address::from_bytes(arg.get(self)?)?;
                let InternedClosure(closure_owner, _) = self.db.lookup_intern_closure((*id).into());
                let infer = self.db.infer(closure_owner);
                let (captures, _) = infer.closure_info(id);
                let layout = self.layout(&self_ty)?;
                let ty_iter = captures.iter().map(|c| c.ty(subst));
                self.exec_clone_for_fields(ty_iter, layout, addr, def, locals, destination, span)?;
            }
            TyKind::Tuple(_, subst) => {
                let [arg] = args else {
                    not_supported!("wrong arg count for clone");
                };
                let addr = Address::from_bytes(arg.get(self)?)?;
                let layout = self.layout(&self_ty)?;
                let ty_iter = subst.iter(Interner).map(|ga| ga.assert_ty_ref(Interner).clone());
                self.exec_clone_for_fields(ty_iter, layout, addr, def, locals, destination, span)?;
            }
            _ => {
                self.exec_fn_with_args(
                    def,
                    args,
                    Substitution::from1(Interner, self_ty),
                    locals,
                    destination,
                    None,
                    span,
                )?;
            }
        }
        Ok(())
    }

    fn exec_clone_for_fields(
        &mut self,
        ty_iter: impl Iterator<Item = Ty>,
        layout: Arc<Layout>,
        addr: Address,
        def: FunctionId,
        locals: &Locals,
        destination: Interval,
        span: MirSpan,
    ) -> Result<()> {
        for (i, ty) in ty_iter.enumerate() {
            let size = self.layout(&ty)?.size.bytes_usize();
            let tmp = self.heap_allocate(self.ptr_size(), self.ptr_size())?;
            let arg = IntervalAndTy {
                interval: Interval { addr: tmp, size: self.ptr_size() },
                ty: TyKind::Ref(Mutability::Not, error_lifetime(), ty.clone()).intern(Interner),
            };
            let offset = layout.fields.offset(i).bytes_usize();
            self.write_memory(tmp, &addr.offset(offset).to_bytes())?;
            self.exec_clone(
                def,
                &[arg],
                ty,
                locals,
                destination.slice(offset..offset + size),
                span,
            )?;
        }
        Ok(())
    }

    fn exec_alloc_fn(
        &mut self,
        alloc_fn: &Symbol,
        args: &[IntervalAndTy],
        destination: Interval,
    ) -> Result<()> {
        match alloc_fn {
            _ if *alloc_fn == sym::rustc_allocator_zeroed || *alloc_fn == sym::rustc_allocator => {
                let [size, align] = args else {
                    return Err(MirEvalError::InternalError(
                        "rustc_allocator args are not provided".into(),
                    ));
                };
                let size = from_bytes!(usize, size.get(self)?);
                let align = from_bytes!(usize, align.get(self)?);
                let result = self.heap_allocate(size, align)?;
                destination.write_from_bytes(self, &result.to_bytes())?;
            }
            _ if *alloc_fn == sym::rustc_deallocator => { /* no-op for now */ }
            _ if *alloc_fn == sym::rustc_reallocator => {
                let [ptr, old_size, align, new_size] = args else {
                    return Err(MirEvalError::InternalError(
                        "rustc_allocator args are not provided".into(),
                    ));
                };
                let old_size = from_bytes!(usize, old_size.get(self)?);
                let new_size = from_bytes!(usize, new_size.get(self)?);
                if old_size >= new_size {
                    destination.write_from_interval(self, ptr.interval)?;
                } else {
                    let ptr = Address::from_bytes(ptr.get(self)?)?;
                    let align = from_bytes!(usize, align.get(self)?);
                    let result = self.heap_allocate(new_size, align)?;
                    Interval { addr: result, size: old_size }
                        .write_from_interval(self, Interval { addr: ptr, size: old_size })?;
                    destination.write_from_bytes(self, &result.to_bytes())?;
                }
            }
            _ => not_supported!("unknown alloc function"),
        }
        Ok(())
    }

    fn detect_lang_function(&self, def: FunctionId) -> Option<LangItem> {
        use LangItem::*;
        let attrs = self.db.attrs(def.into());

        if attrs.by_key(&sym::rustc_const_panic_str).exists() {
            // `#[rustc_const_panic_str]` is treated like `lang = "begin_panic"` by rustc CTFE.
            return Some(LangItem::BeginPanic);
        }

        let candidate = attrs.lang_item()?;
        // We want to execute these functions with special logic
        // `PanicFmt` is not detected here as it's redirected later.
        if [BeginPanic, SliceLen, DropInPlace].contains(&candidate) {
            return Some(candidate);
        }

        None
    }

    fn exec_lang_item(
        &mut self,
        it: LangItem,
        generic_args: &Substitution,
        args: &[IntervalAndTy],
        locals: &Locals,
        span: MirSpan,
    ) -> Result<Vec<u8>> {
        use LangItem::*;
        let mut args = args.iter();
        match it {
            BeginPanic => {
                let mut arg = args
                    .next()
                    .ok_or(MirEvalError::InternalError(
                        "argument of BeginPanic is not provided".into(),
                    ))?
                    .clone();
                while let TyKind::Ref(_, _, ty) = arg.ty.kind(Interner) {
                    if ty.is_str() {
                        let (pointee, metadata) = arg.interval.get(self)?.split_at(self.ptr_size());
                        let len = from_bytes!(usize, metadata);

                        return {
                            Err(MirEvalError::Panic(
                                std::str::from_utf8(
                                    self.read_memory(Address::from_bytes(pointee)?, len)?,
                                )
                                .unwrap()
                                .to_owned(),
                            ))
                        };
                    }
                    let size = self.size_of_sized(ty, locals, "begin panic arg")?;
                    let pointee = arg.interval.get(self)?;
                    arg = IntervalAndTy {
                        interval: Interval::new(Address::from_bytes(pointee)?, size),
                        ty: ty.clone(),
                    };
                }
                Err(MirEvalError::Panic(format!(
                    "unknown-panic-payload: {:?}",
                    arg.ty.kind(Interner)
                )))
            }
            SliceLen => {
                let arg = args.next().ok_or(MirEvalError::InternalError(
                    "argument of <[T]>::len() is not provided".into(),
                ))?;
                let arg = arg.get(self)?;
                let ptr_size = arg.len() / 2;
                Ok(arg[ptr_size..].into())
            }
            DropInPlace => {
                let ty =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner)).ok_or(
                        MirEvalError::InternalError(
                            "generic argument of drop_in_place is not provided".into(),
                        ),
                    )?;
                let arg = args.next().ok_or(MirEvalError::InternalError(
                    "argument of drop_in_place is not provided".into(),
                ))?;
                let arg = arg.interval.get(self)?.to_owned();
                self.run_drop_glue_deep(
                    ty.clone(),
                    locals,
                    Address::from_bytes(&arg[0..self.ptr_size()])?,
                    &arg[self.ptr_size()..],
                    span,
                )?;
                Ok(vec![])
            }
            it => not_supported!("Executing lang item {it:?}"),
        }
    }

    fn exec_syscall(
        &mut self,
        id: i64,
        args: &[IntervalAndTy],
        destination: Interval,
        _locals: &Locals,
        _span: MirSpan,
    ) -> Result<()> {
        match id {
            318 => {
                // SYS_getrandom
                let [buf, len, _flags] = args else {
                    return Err(MirEvalError::InternalError(
                        "SYS_getrandom args are not provided".into(),
                    ));
                };
                let addr = Address::from_bytes(buf.get(self)?)?;
                let size = from_bytes!(usize, len.get(self)?);
                for i in 0..size {
                    let rand_byte = self.random_state.rand_u64() as u8;
                    self.write_memory(addr.offset(i), &[rand_byte])?;
                }
                destination.write_from_interval(self, len.interval)
            }
            _ => {
                not_supported!("Unknown syscall id {id:?}")
            }
        }
    }

    fn exec_extern_c(
        &mut self,
        as_str: &str,
        args: &[IntervalAndTy],
        _generic_args: &Substitution,
        destination: Interval,
        locals: &Locals,
        span: MirSpan,
    ) -> Result<()> {
        match as_str {
            "memcmp" => {
                let [ptr1, ptr2, size] = args else {
                    return Err(MirEvalError::InternalError("memcmp args are not provided".into()));
                };
                let addr1 = Address::from_bytes(ptr1.get(self)?)?;
                let addr2 = Address::from_bytes(ptr2.get(self)?)?;
                let size = from_bytes!(usize, size.get(self)?);
                let slice1 = self.read_memory(addr1, size)?;
                let slice2 = self.read_memory(addr2, size)?;
                let r: i128 = match slice1.cmp(slice2) {
                    cmp::Ordering::Less => -1,
                    cmp::Ordering::Equal => 0,
                    cmp::Ordering::Greater => 1,
                };
                destination.write_from_bytes(self, &r.to_le_bytes()[..destination.size])
            }
            "write" => {
                let [fd, ptr, len] = args else {
                    return Err(MirEvalError::InternalError(
                        "libc::write args are not provided".into(),
                    ));
                };
                let fd = u128::from_le_bytes(pad16(fd.get(self)?, false));
                let interval = Interval {
                    addr: Address::from_bytes(ptr.get(self)?)?,
                    size: from_bytes!(usize, len.get(self)?),
                };
                match fd {
                    1 => {
                        self.write_to_stdout(interval)?;
                    }
                    2 => {
                        self.write_to_stderr(interval)?;
                    }
                    _ => not_supported!("write to arbitrary file descriptor"),
                }
                destination.write_from_interval(self, len.interval)?;
                Ok(())
            }
            "pthread_key_create" => {
                let key = self.thread_local_storage.create_key();
                let Some(arg0) = args.first() else {
                    return Err(MirEvalError::InternalError(
                        "pthread_key_create arg0 is not provided".into(),
                    ));
                };
                let arg0_addr = Address::from_bytes(arg0.get(self)?)?;
                let key_ty = if let Some((ty, ..)) = arg0.ty.as_reference_or_ptr() {
                    ty
                } else {
                    return Err(MirEvalError::InternalError(
                        "pthread_key_create arg0 is not a pointer".into(),
                    ));
                };
                let arg0_interval = Interval::new(
                    arg0_addr,
                    self.size_of_sized(key_ty, locals, "pthread_key_create key arg")?,
                );
                arg0_interval.write_from_bytes(self, &key.to_le_bytes()[0..arg0_interval.size])?;
                // return 0 as success
                destination.write_from_bytes(self, &0u64.to_le_bytes()[0..destination.size])?;
                Ok(())
            }
            "pthread_getspecific" => {
                let Some(arg0) = args.first() else {
                    return Err(MirEvalError::InternalError(
                        "pthread_getspecific arg0 is not provided".into(),
                    ));
                };
                let key = from_bytes!(usize, &pad16(arg0.get(self)?, false)[0..8]);
                let value = self.thread_local_storage.get_key(key)?;
                destination.write_from_bytes(self, &value.to_le_bytes()[0..destination.size])?;
                Ok(())
            }
            "pthread_setspecific" => {
                let Some(arg0) = args.first() else {
                    return Err(MirEvalError::InternalError(
                        "pthread_setspecific arg0 is not provided".into(),
                    ));
                };
                let key = from_bytes!(usize, &pad16(arg0.get(self)?, false)[0..8]);
                let Some(arg1) = args.get(1) else {
                    return Err(MirEvalError::InternalError(
                        "pthread_setspecific arg1 is not provided".into(),
                    ));
                };
                let value = from_bytes!(u128, pad16(arg1.get(self)?, false));
                self.thread_local_storage.set_key(key, value)?;
                // return 0 as success
                destination.write_from_bytes(self, &0u64.to_le_bytes()[0..destination.size])?;
                Ok(())
            }
            "pthread_key_delete" => {
                // we ignore this currently
                // return 0 as success
                destination.write_from_bytes(self, &0u64.to_le_bytes()[0..destination.size])?;
                Ok(())
            }
            "syscall" => {
                let Some((id, rest)) = args.split_first() else {
                    return Err(MirEvalError::InternalError("syscall arg1 is not provided".into()));
                };
                let id = from_bytes!(i64, id.get(self)?);
                self.exec_syscall(id, rest, destination, locals, span)
            }
            "sched_getaffinity" => {
                let [_pid, _set_size, set] = args else {
                    return Err(MirEvalError::InternalError(
                        "libc::write args are not provided".into(),
                    ));
                };
                let set = Address::from_bytes(set.get(self)?)?;
                // Only enable core 0 (we are single threaded anyway), which is bitset 0x0000001
                self.write_memory(set, &[1])?;
                // return 0 as success
                self.write_memory_using_ref(destination.addr, destination.size)?.fill(0);
                Ok(())
            }
            "getenv" => {
                let [name] = args else {
                    return Err(MirEvalError::InternalError(
                        "libc::write args are not provided".into(),
                    ));
                };
                let mut name_buf = vec![];
                let name = {
                    let mut index = Address::from_bytes(name.get(self)?)?;
                    loop {
                        let byte = self.read_memory(index, 1)?[0];
                        index = index.offset(1);
                        if byte == 0 {
                            break;
                        }
                        name_buf.push(byte);
                    }
                    String::from_utf8_lossy(&name_buf)
                };
                let value = self.db.crate_graph()[self.crate_id].env.get(&name);
                match value {
                    None => {
                        // Write null as fail
                        self.write_memory_using_ref(destination.addr, destination.size)?.fill(0);
                    }
                    Some(mut value) => {
                        value.push('\0');
                        let addr = self.heap_allocate(value.len(), 1)?;
                        self.write_memory(addr, value.as_bytes())?;
                        self.write_memory(destination.addr, &addr.to_bytes())?;
                    }
                }
                Ok(())
            }
            _ => not_supported!("unknown external function {as_str}"),
        }
    }

    fn exec_intrinsic(
        &mut self,
        name: &str,
        args: &[IntervalAndTy],
        generic_args: &Substitution,
        destination: Interval,
        locals: &Locals,
        span: MirSpan,
        needs_override: bool,
    ) -> Result<bool> {
        if let Some(name) = name.strip_prefix("atomic_") {
            return self
                .exec_atomic_intrinsic(name, args, generic_args, destination, locals, span)
                .map(|()| true);
        }
        if let Some(name) = name.strip_prefix("simd_") {
            return self
                .exec_simd_intrinsic(name, args, generic_args, destination, locals, span)
                .map(|()| true);
        }
        // FIXME(#17451): Add `f16` and `f128` intrinsics.
        if let Some(name) = name.strip_suffix("f64") {
            let result = match name {
                "sqrt" | "sin" | "cos" | "exp" | "exp2" | "log" | "log10" | "log2" | "fabs"
                | "floor" | "ceil" | "trunc" | "rint" | "nearbyint" | "round" | "roundeven" => {
                    let [arg] = args else {
                        return Err(MirEvalError::InternalError(
                            "f64 intrinsic signature doesn't match fn (f64) -> f64".into(),
                        ));
                    };
                    let arg = from_bytes!(f64, arg.get(self)?);
                    match name {
                        "sqrt" => arg.sqrt(),
                        "sin" => arg.sin(),
                        "cos" => arg.cos(),
                        "exp" => arg.exp(),
                        "exp2" => arg.exp2(),
                        "log" => arg.ln(),
                        "log10" => arg.log10(),
                        "log2" => arg.log2(),
                        "fabs" => arg.abs(),
                        "floor" => arg.floor(),
                        "ceil" => arg.ceil(),
                        "trunc" => arg.trunc(),
                        // FIXME: these rounds should be different, but only `.round()` is stable now.
                        "rint" => arg.round(),
                        "nearbyint" => arg.round(),
                        "round" => arg.round(),
                        "roundeven" => arg.round(),
                        _ => unreachable!(),
                    }
                }
                "pow" | "minnum" | "maxnum" | "copysign" => {
                    let [arg1, arg2] = args else {
                        return Err(MirEvalError::InternalError(
                            "f64 intrinsic signature doesn't match fn (f64, f64) -> f64".into(),
                        ));
                    };
                    let arg1 = from_bytes!(f64, arg1.get(self)?);
                    let arg2 = from_bytes!(f64, arg2.get(self)?);
                    match name {
                        "pow" => arg1.powf(arg2),
                        "minnum" => arg1.min(arg2),
                        "maxnum" => arg1.max(arg2),
                        "copysign" => arg1.copysign(arg2),
                        _ => unreachable!(),
                    }
                }
                "powi" => {
                    let [arg1, arg2] = args else {
                        return Err(MirEvalError::InternalError(
                            "powif64 signature doesn't match fn (f64, i32) -> f64".into(),
                        ));
                    };
                    let arg1 = from_bytes!(f64, arg1.get(self)?);
                    let arg2 = from_bytes!(i32, arg2.get(self)?);
                    arg1.powi(arg2)
                }
                "fma" => {
                    let [arg1, arg2, arg3] = args else {
                        return Err(MirEvalError::InternalError(
                            "fmaf64 signature doesn't match fn (f64, f64, f64) -> f64".into(),
                        ));
                    };
                    let arg1 = from_bytes!(f64, arg1.get(self)?);
                    let arg2 = from_bytes!(f64, arg2.get(self)?);
                    let arg3 = from_bytes!(f64, arg3.get(self)?);
                    arg1.mul_add(arg2, arg3)
                }
                _ => not_supported!("unknown f64 intrinsic {name}"),
            };
            return destination.write_from_bytes(self, &result.to_le_bytes()).map(|()| true);
        }
        if let Some(name) = name.strip_suffix("f32") {
            let result = match name {
                "sqrt" | "sin" | "cos" | "exp" | "exp2" | "log" | "log10" | "log2" | "fabs"
                | "floor" | "ceil" | "trunc" | "rint" | "nearbyint" | "round" | "roundeven" => {
                    let [arg] = args else {
                        return Err(MirEvalError::InternalError(
                            "f32 intrinsic signature doesn't match fn (f32) -> f32".into(),
                        ));
                    };
                    let arg = from_bytes!(f32, arg.get(self)?);
                    match name {
                        "sqrt" => arg.sqrt(),
                        "sin" => arg.sin(),
                        "cos" => arg.cos(),
                        "exp" => arg.exp(),
                        "exp2" => arg.exp2(),
                        "log" => arg.ln(),
                        "log10" => arg.log10(),
                        "log2" => arg.log2(),
                        "fabs" => arg.abs(),
                        "floor" => arg.floor(),
                        "ceil" => arg.ceil(),
                        "trunc" => arg.trunc(),
                        // FIXME: these rounds should be different, but only `.round()` is stable now.
                        "rint" => arg.round(),
                        "nearbyint" => arg.round(),
                        "round" => arg.round(),
                        "roundeven" => arg.round(),
                        _ => unreachable!(),
                    }
                }
                "pow" | "minnum" | "maxnum" | "copysign" => {
                    let [arg1, arg2] = args else {
                        return Err(MirEvalError::InternalError(
                            "f32 intrinsic signature doesn't match fn (f32, f32) -> f32".into(),
                        ));
                    };
                    let arg1 = from_bytes!(f32, arg1.get(self)?);
                    let arg2 = from_bytes!(f32, arg2.get(self)?);
                    match name {
                        "pow" => arg1.powf(arg2),
                        "minnum" => arg1.min(arg2),
                        "maxnum" => arg1.max(arg2),
                        "copysign" => arg1.copysign(arg2),
                        _ => unreachable!(),
                    }
                }
                "powi" => {
                    let [arg1, arg2] = args else {
                        return Err(MirEvalError::InternalError(
                            "powif32 signature doesn't match fn (f32, i32) -> f32".into(),
                        ));
                    };
                    let arg1 = from_bytes!(f32, arg1.get(self)?);
                    let arg2 = from_bytes!(i32, arg2.get(self)?);
                    arg1.powi(arg2)
                }
                "fma" => {
                    let [arg1, arg2, arg3] = args else {
                        return Err(MirEvalError::InternalError(
                            "fmaf32 signature doesn't match fn (f32, f32, f32) -> f32".into(),
                        ));
                    };
                    let arg1 = from_bytes!(f32, arg1.get(self)?);
                    let arg2 = from_bytes!(f32, arg2.get(self)?);
                    let arg3 = from_bytes!(f32, arg3.get(self)?);
                    arg1.mul_add(arg2, arg3)
                }
                _ => not_supported!("unknown f32 intrinsic {name}"),
            };
            return destination.write_from_bytes(self, &result.to_le_bytes()).map(|()| true);
        }
        match name {
            "size_of" => {
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "size_of generic arg is not provided".into(),
                    ));
                };
                let size = self.size_of_sized(ty, locals, "size_of arg")?;
                destination.write_from_bytes(self, &size.to_le_bytes()[0..destination.size])
            }
            "min_align_of" | "pref_align_of" => {
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "align_of generic arg is not provided".into(),
                    ));
                };
                let align = self.layout(ty)?.align.abi.bytes();
                destination.write_from_bytes(self, &align.to_le_bytes()[0..destination.size])
            }
            "size_of_val" => {
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "size_of_val generic arg is not provided".into(),
                    ));
                };
                let [arg] = args else {
                    return Err(MirEvalError::InternalError(
                        "size_of_val args are not provided".into(),
                    ));
                };
                if let Some((size, _)) = self.size_align_of(ty, locals)? {
                    destination.write_from_bytes(self, &size.to_le_bytes())
                } else {
                    let metadata = arg.interval.slice(self.ptr_size()..self.ptr_size() * 2);
                    let (size, _) = self.size_align_of_unsized(ty, metadata, locals)?;
                    destination.write_from_bytes(self, &size.to_le_bytes())
                }
            }
            "min_align_of_val" => {
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "min_align_of_val generic arg is not provided".into(),
                    ));
                };
                let [arg] = args else {
                    return Err(MirEvalError::InternalError(
                        "min_align_of_val args are not provided".into(),
                    ));
                };
                if let Some((_, align)) = self.size_align_of(ty, locals)? {
                    destination.write_from_bytes(self, &align.to_le_bytes())
                } else {
                    let metadata = arg.interval.slice(self.ptr_size()..self.ptr_size() * 2);
                    let (_, align) = self.size_align_of_unsized(ty, metadata, locals)?;
                    destination.write_from_bytes(self, &align.to_le_bytes())
                }
            }
            "type_name" => {
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "type_name generic arg is not provided".into(),
                    ));
                };
                let ty_name = match ty.display_source_code(
                    self.db,
                    locals.body.owner.module(self.db.upcast()),
                    true,
                ) {
                    Ok(ty_name) => ty_name,
                    // Fallback to human readable display in case of `Err`. Ideally we want to use `display_source_code` to
                    // render full paths.
                    Err(_) => {
                        let krate = locals.body.owner.krate(self.db.upcast());
                        ty.display(self.db, DisplayTarget::from_crate(self.db, krate)).to_string()
                    }
                };
                let len = ty_name.len();
                let addr = self.heap_allocate(len, 1)?;
                self.write_memory(addr, ty_name.as_bytes())?;
                destination.slice(0..self.ptr_size()).write_from_bytes(self, &addr.to_bytes())?;
                destination
                    .slice(self.ptr_size()..2 * self.ptr_size())
                    .write_from_bytes(self, &len.to_le_bytes())
            }
            "needs_drop" => {
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "size_of generic arg is not provided".into(),
                    ));
                };
                let result = match self.db.has_drop_glue(ty.clone(), self.trait_env.clone()) {
                    DropGlue::HasDropGlue => true,
                    DropGlue::None => false,
                    DropGlue::DependOnParams => {
                        never!("should be fully monomorphized now");
                        true
                    }
                };
                destination.write_from_bytes(self, &[u8::from(result)])
            }
            "ptr_guaranteed_cmp" => {
                // FIXME: this is wrong for const eval, it should return 2 in some
                // cases.
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "wrapping_add args are not provided".into(),
                    ));
                };
                let ans = lhs.get(self)? == rhs.get(self)?;
                destination.write_from_bytes(self, &[u8::from(ans)])
            }
            "saturating_add" | "saturating_sub" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "saturating_add args are not provided".into(),
                    ));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = match name {
                    "saturating_add" => lhs.saturating_add(rhs),
                    "saturating_sub" => lhs.saturating_sub(rhs),
                    _ => unreachable!(),
                };
                let bits = destination.size * 8;
                // FIXME: signed
                let is_signed = false;
                let mx: u128 = if is_signed { (1 << (bits - 1)) - 1 } else { (1 << bits) - 1 };
                // FIXME: signed
                let mn: u128 = 0;
                let ans = cmp::min(mx, cmp::max(mn, ans));
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "wrapping_add" | "unchecked_add" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "wrapping_add args are not provided".into(),
                    ));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_add(rhs);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "ptr_offset_from_unsigned" | "ptr_offset_from" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "wrapping_sub args are not provided".into(),
                    ));
                };
                let lhs = i128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = i128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_sub(rhs);
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "ptr_offset_from generic arg is not provided".into(),
                    ));
                };
                let size = self.size_of_sized(ty, locals, "ptr_offset_from arg")? as i128;
                let ans = ans / size;
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "wrapping_sub" | "unchecked_sub" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "wrapping_sub args are not provided".into(),
                    ));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_sub(rhs);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "wrapping_mul" | "unchecked_mul" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "wrapping_mul args are not provided".into(),
                    ));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_mul(rhs);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "wrapping_shl" | "unchecked_shl" => {
                // FIXME: signed
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "unchecked_shl args are not provided".into(),
                    ));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_shl(rhs as u32);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "wrapping_shr" | "unchecked_shr" => {
                // FIXME: signed
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "unchecked_shr args are not provided".into(),
                    ));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_shr(rhs as u32);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "unchecked_rem" => {
                // FIXME: signed
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "unchecked_rem args are not provided".into(),
                    ));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.checked_rem(rhs).ok_or_else(|| {
                    MirEvalError::UndefinedBehavior("unchecked_rem with bad inputs".to_owned())
                })?;
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "unchecked_div" | "exact_div" => {
                // FIXME: signed
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "unchecked_div args are not provided".into(),
                    ));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.checked_div(rhs).ok_or_else(|| {
                    MirEvalError::UndefinedBehavior("unchecked_rem with bad inputs".to_owned())
                })?;
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "add_with_overflow" | "sub_with_overflow" | "mul_with_overflow" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "const_eval_select args are not provided".into(),
                    ));
                };
                let result_ty = TyKind::Tuple(
                    2,
                    Substitution::from_iter(Interner, [lhs.ty.clone(), TyBuilder::bool()]),
                )
                .intern(Interner);
                let op_size =
                    self.size_of_sized(&lhs.ty, locals, "operand of add_with_overflow")?;
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let (ans, u128overflow) = match name {
                    "add_with_overflow" => lhs.overflowing_add(rhs),
                    "sub_with_overflow" => lhs.overflowing_sub(rhs),
                    "mul_with_overflow" => lhs.overflowing_mul(rhs),
                    _ => unreachable!(),
                };
                let is_overflow = u128overflow
                    || ans.to_le_bytes()[op_size..].iter().any(|&it| it != 0 && it != 255);
                let is_overflow = vec![u8::from(is_overflow)];
                let layout = self.layout(&result_ty)?;
                let result = self.construct_with_layout(
                    layout.size.bytes_usize(),
                    &layout,
                    None,
                    [ans.to_le_bytes()[0..op_size].to_vec(), is_overflow]
                        .into_iter()
                        .map(IntervalOrOwned::Owned),
                )?;
                destination.write_from_bytes(self, &result)
            }
            "copy" | "copy_nonoverlapping" => {
                let [src, dst, offset] = args else {
                    return Err(MirEvalError::InternalError(
                        "copy_nonoverlapping args are not provided".into(),
                    ));
                };
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "copy_nonoverlapping generic arg is not provided".into(),
                    ));
                };
                let src = Address::from_bytes(src.get(self)?)?;
                let dst = Address::from_bytes(dst.get(self)?)?;
                let offset = from_bytes!(usize, offset.get(self)?);
                let size = self.size_of_sized(ty, locals, "copy_nonoverlapping ptr type")?;
                let size = offset * size;
                let src = Interval { addr: src, size };
                let dst = Interval { addr: dst, size };
                dst.write_from_interval(self, src)
            }
            "offset" | "arith_offset" => {
                let [ptr, offset] = args else {
                    return Err(MirEvalError::InternalError("offset args are not provided".into()));
                };
                let ty = if name == "offset" {
                    let Some(ty0) =
                        generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                    else {
                        return Err(MirEvalError::InternalError(
                            "offset generic arg is not provided".into(),
                        ));
                    };
                    let Some(ty1) =
                        generic_args.as_slice(Interner).get(1).and_then(|it| it.ty(Interner))
                    else {
                        return Err(MirEvalError::InternalError(
                            "offset generic arg is not provided".into(),
                        ));
                    };
                    if !matches!(
                        ty1.as_builtin(),
                        Some(
                            BuiltinType::Int(BuiltinInt::Isize)
                                | BuiltinType::Uint(BuiltinUint::Usize)
                        )
                    ) {
                        return Err(MirEvalError::InternalError(
                            "offset generic arg is not usize or isize".into(),
                        ));
                    }
                    match ty0.as_raw_ptr() {
                        Some((ty, _)) => ty,
                        None => {
                            return Err(MirEvalError::InternalError(
                                "offset generic arg is not a raw pointer".into(),
                            ));
                        }
                    }
                } else {
                    let Some(ty) =
                        generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                    else {
                        return Err(MirEvalError::InternalError(
                            "arith_offset generic arg is not provided".into(),
                        ));
                    };
                    ty
                };
                let ptr = u128::from_le_bytes(pad16(ptr.get(self)?, false));
                let offset = u128::from_le_bytes(pad16(offset.get(self)?, false));
                let size = self.size_of_sized(ty, locals, "offset ptr type")? as u128;
                let ans = ptr + offset * size;
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "assert_inhabited" | "assert_zero_valid" | "assert_uninit_valid" | "assume" => {
                // FIXME: We should actually implement these checks
                Ok(())
            }
            "forget" => {
                // We don't call any drop glue yet, so there is nothing here
                Ok(())
            }
            "transmute" => {
                let [arg] = args else {
                    return Err(MirEvalError::InternalError(
                        "transmute arg is not provided".into(),
                    ));
                };
                destination.write_from_interval(self, arg.interval)
            }
            "ctpop" => {
                let [arg] = args else {
                    return Err(MirEvalError::InternalError("ctpop arg is not provided".into()));
                };
                let result = u128::from_le_bytes(pad16(arg.get(self)?, false)).count_ones();
                destination
                    .write_from_bytes(self, &(result as u128).to_le_bytes()[0..destination.size])
            }
            "ctlz" | "ctlz_nonzero" => {
                let [arg] = args else {
                    return Err(MirEvalError::InternalError("ctlz arg is not provided".into()));
                };
                let result =
                    u128::from_le_bytes(pad16(arg.get(self)?, false)).leading_zeros() as usize;
                let result = result - (128 - arg.interval.size * 8);
                destination
                    .write_from_bytes(self, &(result as u128).to_le_bytes()[0..destination.size])
            }
            "cttz" | "cttz_nonzero" => {
                let [arg] = args else {
                    return Err(MirEvalError::InternalError("cttz arg is not provided".into()));
                };
                let result = u128::from_le_bytes(pad16(arg.get(self)?, false)).trailing_zeros();
                destination
                    .write_from_bytes(self, &(result as u128).to_le_bytes()[0..destination.size])
            }
            "rotate_left" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "rotate_left args are not provided".into(),
                    ));
                };
                let lhs = &lhs.get(self)?[0..destination.size];
                let rhs = rhs.get(self)?[0] as u32;
                match destination.size {
                    1 => {
                        let r = from_bytes!(u8, lhs).rotate_left(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    2 => {
                        let r = from_bytes!(u16, lhs).rotate_left(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    4 => {
                        let r = from_bytes!(u32, lhs).rotate_left(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    8 => {
                        let r = from_bytes!(u64, lhs).rotate_left(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    16 => {
                        let r = from_bytes!(u128, lhs).rotate_left(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    s => not_supported!("destination with size {s} for rotate_left"),
                }
            }
            "rotate_right" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "rotate_right args are not provided".into(),
                    ));
                };
                let lhs = &lhs.get(self)?[0..destination.size];
                let rhs = rhs.get(self)?[0] as u32;
                match destination.size {
                    1 => {
                        let r = from_bytes!(u8, lhs).rotate_right(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    2 => {
                        let r = from_bytes!(u16, lhs).rotate_right(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    4 => {
                        let r = from_bytes!(u32, lhs).rotate_right(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    8 => {
                        let r = from_bytes!(u64, lhs).rotate_right(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    16 => {
                        let r = from_bytes!(u128, lhs).rotate_right(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    s => not_supported!("destination with size {s} for rotate_right"),
                }
            }
            "discriminant_value" => {
                let [arg] = args else {
                    return Err(MirEvalError::InternalError(
                        "discriminant_value arg is not provided".into(),
                    ));
                };
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "discriminant_value generic arg is not provided".into(),
                    ));
                };
                let addr = Address::from_bytes(arg.get(self)?)?;
                let size = self.size_of_sized(ty, locals, "discriminant_value ptr type")?;
                let interval = Interval { addr, size };
                let r = self.compute_discriminant(ty.clone(), interval.get(self)?)?;
                destination.write_from_bytes(self, &r.to_le_bytes()[0..destination.size])
            }
            "const_eval_select" => {
                let [tuple, const_fn, _] = args else {
                    return Err(MirEvalError::InternalError(
                        "const_eval_select args are not provided".into(),
                    ));
                };
                let mut args = vec![const_fn.clone()];
                let TyKind::Tuple(_, fields) = tuple.ty.kind(Interner) else {
                    return Err(MirEvalError::InternalError(
                        "const_eval_select arg[0] is not a tuple".into(),
                    ));
                };
                let layout = self.layout(&tuple.ty)?;
                for (i, field) in fields.iter(Interner).enumerate() {
                    let field = field.assert_ty_ref(Interner).clone();
                    let offset = layout.fields.offset(i).bytes_usize();
                    let addr = tuple.interval.addr.offset(offset);
                    args.push(IntervalAndTy::new(addr, field, self, locals)?);
                }
                if let Some(target) = self.db.lang_item(self.crate_id, LangItem::FnOnce) {
                    if let Some(def) = target.as_trait().and_then(|it| {
                        self.db
                            .trait_data(it)
                            .method_by_name(&Name::new_symbol_root(sym::call_once.clone()))
                    }) {
                        self.exec_fn_trait(
                            def,
                            &args,
                            // FIXME: wrong for manual impls of `FnOnce`
                            Substitution::empty(Interner),
                            locals,
                            destination,
                            None,
                            span,
                        )?;
                        return Ok(true);
                    }
                }
                not_supported!("FnOnce was not available for executing const_eval_select");
            }
            "read_via_copy" | "volatile_load" => {
                let [arg] = args else {
                    return Err(MirEvalError::InternalError(
                        "read_via_copy args are not provided".into(),
                    ));
                };
                let addr = Address::from_bytes(arg.interval.get(self)?)?;
                destination.write_from_interval(self, Interval { addr, size: destination.size })
            }
            "write_via_move" => {
                let [ptr, val] = args else {
                    return Err(MirEvalError::InternalError(
                        "write_via_move args are not provided".into(),
                    ));
                };
                let dst = Address::from_bytes(ptr.get(self)?)?;
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "write_via_copy generic arg is not provided".into(),
                    ));
                };
                let size = self.size_of_sized(ty, locals, "write_via_move ptr type")?;
                Interval { addr: dst, size }.write_from_interval(self, val.interval)?;
                Ok(())
            }
            "write_bytes" => {
                let [dst, val, count] = args else {
                    return Err(MirEvalError::InternalError(
                        "write_bytes args are not provided".into(),
                    ));
                };
                let count = from_bytes!(usize, count.get(self)?);
                let val = from_bytes!(u8, val.get(self)?);
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "write_bytes generic arg is not provided".into(),
                    ));
                };
                let dst = Address::from_bytes(dst.get(self)?)?;
                let size = self.size_of_sized(ty, locals, "copy_nonoverlapping ptr type")?;
                let size = count * size;
                self.write_memory_using_ref(dst, size)?.fill(val);
                Ok(())
            }
            "ptr_metadata" => {
                let [ptr] = args else {
                    return Err(MirEvalError::InternalError(
                        "ptr_metadata args are not provided".into(),
                    ));
                };
                let arg = ptr.interval.get(self)?.to_owned();
                let metadata = &arg[self.ptr_size()..];
                destination.write_from_bytes(self, metadata)?;
                Ok(())
            }
            "three_way_compare" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::InternalError(
                        "three_way_compare args are not provided".into(),
                    ));
                };
                let Some(ty) =
                    generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::InternalError(
                        "three_way_compare generic arg is not provided".into(),
                    ));
                };
                let signed = match ty.as_builtin().unwrap() {
                    BuiltinType::Int(_) => true,
                    BuiltinType::Uint(_) => false,
                    _ => {
                        return Err(MirEvalError::InternalError(
                            "three_way_compare expects an integral type".into(),
                        ))
                    }
                };
                let rhs = rhs.get(self)?;
                let lhs = lhs.get(self)?;
                let mut result = Ordering::Equal;
                for (l, r) in lhs.iter().zip(rhs).rev() {
                    let it = l.cmp(r);
                    if it != Ordering::Equal {
                        result = it;
                        break;
                    }
                }
                if signed {
                    if let Some((&l, &r)) = lhs.iter().zip(rhs).next_back() {
                        if l != r {
                            result = (l as i8).cmp(&(r as i8));
                        }
                    }
                }
                if let Some(LangItemTarget::EnumId(e)) =
                    self.db.lang_item(self.crate_id, LangItem::Ordering)
                {
                    let ty = self.db.ty(e.into());
                    let r = self
                        .compute_discriminant(ty.skip_binders().clone(), &[result as i8 as u8])?;
                    destination.write_from_bytes(self, &r.to_le_bytes()[0..destination.size])?;
                    Ok(())
                } else {
                    Err(MirEvalError::InternalError("Ordering enum not found".into()))
                }
            }
            "aggregate_raw_ptr" => {
                let [data, meta] = args else {
                    return Err(MirEvalError::InternalError(
                        "aggregate_raw_ptr args are not provided".into(),
                    ));
                };
                destination.write_from_interval(self, data.interval)?;
                Interval {
                    addr: destination.addr.offset(data.interval.size),
                    size: destination.size - data.interval.size,
                }
                .write_from_interval(self, meta.interval)?;
                Ok(())
            }
            _ if needs_override => not_supported!("intrinsic {name} is not implemented"),
            _ => return Ok(false),
        }
        .map(|()| true)
    }

    fn size_align_of_unsized(
        &mut self,
        ty: &Ty,
        metadata: Interval,
        locals: &Locals,
    ) -> Result<(usize, usize)> {
        Ok(match ty.kind(Interner) {
            TyKind::Str => (from_bytes!(usize, metadata.get(self)?), 1),
            TyKind::Slice(inner) => {
                let len = from_bytes!(usize, metadata.get(self)?);
                let (size, align) = self.size_align_of_sized(inner, locals, "slice inner type")?;
                (size * len, align)
            }
            TyKind::Dyn(_) => self.size_align_of_sized(
                self.vtable_map.ty_of_bytes(metadata.get(self)?)?,
                locals,
                "dyn concrete type",
            )?,
            TyKind::Adt(id, subst) => {
                let id = id.0;
                let layout = self.layout_adt(id, subst.clone())?;
                let id = match id {
                    AdtId::StructId(s) => s,
                    _ => not_supported!("unsized enum or union"),
                };
                let field_types = &self.db.field_types(id.into());
                let last_field_ty =
                    field_types.iter().next_back().unwrap().1.clone().substitute(Interner, subst);
                let sized_part_size =
                    layout.fields.offset(field_types.iter().count() - 1).bytes_usize();
                let sized_part_align = layout.align.abi.bytes() as usize;
                let (unsized_part_size, unsized_part_align) =
                    self.size_align_of_unsized(&last_field_ty, metadata, locals)?;
                let align = sized_part_align.max(unsized_part_align) as isize;
                let size = (sized_part_size + unsized_part_size) as isize;
                // Must add any necessary padding to `size`
                // (to make it a multiple of `align`) before returning it.
                //
                // Namely, the returned size should be, in C notation:
                //
                //   `size + ((size & (align-1)) ? align : 0)`
                //
                // emulated via the semi-standard fast bit trick:
                //
                //   `(size + (align-1)) & -align`
                let size = (size + (align - 1)) & (-align);
                (size as usize, align as usize)
            }
            _ => not_supported!("unsized type other than str, slice, struct and dyn"),
        })
    }

    fn exec_atomic_intrinsic(
        &mut self,
        name: &str,
        args: &[IntervalAndTy],
        generic_args: &Substitution,
        destination: Interval,
        locals: &Locals,
        _span: MirSpan,
    ) -> Result<()> {
        // We are a single threaded runtime with no UB checking and no optimization, so
        // we can implement atomic intrinsics as normal functions.

        if name.starts_with("singlethreadfence_") || name.starts_with("fence_") {
            return Ok(());
        }

        // The rest of atomic intrinsics have exactly one generic arg

        let Some(ty) = generic_args.as_slice(Interner).first().and_then(|it| it.ty(Interner))
        else {
            return Err(MirEvalError::InternalError(
                "atomic intrinsic generic arg is not provided".into(),
            ));
        };
        let Some(arg0) = args.first() else {
            return Err(MirEvalError::InternalError(
                "atomic intrinsic arg0 is not provided".into(),
            ));
        };
        let arg0_addr = Address::from_bytes(arg0.get(self)?)?;
        let arg0_interval =
            Interval::new(arg0_addr, self.size_of_sized(ty, locals, "atomic intrinsic type arg")?);
        if name.starts_with("load_") {
            return destination.write_from_interval(self, arg0_interval);
        }
        let Some(arg1) = args.get(1) else {
            return Err(MirEvalError::InternalError(
                "atomic intrinsic arg1 is not provided".into(),
            ));
        };
        if name.starts_with("store_") {
            return arg0_interval.write_from_interval(self, arg1.interval);
        }
        if name.starts_with("xchg_") {
            destination.write_from_interval(self, arg0_interval)?;
            return arg0_interval.write_from_interval(self, arg1.interval);
        }
        if name.starts_with("xadd_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = lhs.wrapping_add(rhs);
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        if name.starts_with("xsub_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = lhs.wrapping_sub(rhs);
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        if name.starts_with("and_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = lhs & rhs;
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        if name.starts_with("or_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = lhs | rhs;
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        if name.starts_with("xor_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = lhs ^ rhs;
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        if name.starts_with("nand_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = !(lhs & rhs);
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        let Some(arg2) = args.get(2) else {
            return Err(MirEvalError::InternalError(
                "atomic intrinsic arg2 is not provided".into(),
            ));
        };
        if name.starts_with("cxchg_") || name.starts_with("cxchgweak_") {
            let dest = if arg1.get(self)? == arg0_interval.get(self)? {
                arg0_interval.write_from_interval(self, arg2.interval)?;
                (arg1.interval, true)
            } else {
                (arg0_interval, false)
            };
            let result_ty = TyKind::Tuple(
                2,
                Substitution::from_iter(Interner, [ty.clone(), TyBuilder::bool()]),
            )
            .intern(Interner);
            let layout = self.layout(&result_ty)?;
            let result = self.construct_with_layout(
                layout.size.bytes_usize(),
                &layout,
                None,
                [IntervalOrOwned::Borrowed(dest.0), IntervalOrOwned::Owned(vec![u8::from(dest.1)])]
                    .into_iter(),
            )?;
            return destination.write_from_bytes(self, &result);
        }
        not_supported!("unknown atomic intrinsic {name}");
    }
}
