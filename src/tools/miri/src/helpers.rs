use std::num::NonZero;
use std::time::Duration;
use std::{cmp, iter};

use rand::RngCore;
use rustc_abi::{Align, CanonAbi, ExternAbi, FieldIdx, FieldsShape, Size, Variants};
use rustc_apfloat::Float;
use rustc_apfloat::ieee::{Double, Half, Quad, Single};
use rustc_hir::Safety;
use rustc_hir::def::{DefKind, Namespace};
use rustc_hir::def_id::{CRATE_DEF_INDEX, CrateNum, DefId, LOCAL_CRATE};
use rustc_index::IndexVec;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_middle::middle::exported_symbols::ExportedSymbol;
use rustc_middle::ty::layout::{FnAbiOf, LayoutOf, MaybeResult, TyAndLayout};
use rustc_middle::ty::{self, Binder, FloatTy, FnSig, IntTy, Ty, TyCtxt, UintTy};
use rustc_session::config::CrateType;
use rustc_span::{Span, Symbol};
use rustc_symbol_mangling::mangle_internal_symbol;
use rustc_target::callconv::FnAbi;

use crate::*;

/// Indicates which kind of access is being performed.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum AccessKind {
    Read,
    Write,
}

/// Gets an instance for a path.
///
/// A `None` namespace indicates we are looking for a module.
fn try_resolve_did(tcx: TyCtxt<'_>, path: &[&str], namespace: Option<Namespace>) -> Option<DefId> {
    /// Yield all children of the given item, that have the given name.
    fn find_children<'tcx: 'a, 'a>(
        tcx: TyCtxt<'tcx>,
        item: DefId,
        name: &'a str,
    ) -> impl Iterator<Item = DefId> + 'a {
        let name = Symbol::intern(name);
        tcx.module_children(item)
            .iter()
            .filter(move |item| item.ident.name == name)
            .map(move |item| item.res.def_id())
    }

    // Take apart the path: leading crate, a sequence of modules, and potentially a final item.
    let (&crate_name, path) = path.split_first().expect("paths must have at least one segment");
    let (modules, item) = if let Some(namespace) = namespace {
        let (&item_name, modules) =
            path.split_last().expect("non-module paths must have at least 2 segments");
        (modules, Some((item_name, namespace)))
    } else {
        (path, None)
    };

    // There may be more than one crate with this name. We try them all.
    // (This is particularly relevant when running `std` tests as then there are two `std` crates:
    // the one in the sysroot and the one locally built by `cargo test`.)
    // FIXME: can we prefer the one from the sysroot?
    'crates: for krate in
        tcx.crates(()).iter().filter(|&&krate| tcx.crate_name(krate).as_str() == crate_name)
    {
        let mut cur_item = DefId { krate: *krate, index: CRATE_DEF_INDEX };
        // Go over the modules.
        for &segment in modules {
            let Some(next_item) = find_children(tcx, cur_item, segment)
                .find(|item| tcx.def_kind(item) == DefKind::Mod)
            else {
                continue 'crates;
            };
            cur_item = next_item;
        }
        // Finally, look up the desired item in this module, if any.
        match item {
            Some((item_name, namespace)) => {
                let Some(item) = find_children(tcx, cur_item, item_name)
                    .find(|item| tcx.def_kind(item).ns() == Some(namespace))
                else {
                    continue 'crates;
                };
                return Some(item);
            }
            None => {
                // Just return the module.
                return Some(cur_item);
            }
        }
    }
    // Item not found in any of the crates with the right name.
    None
}

/// Gets an instance for a path; fails gracefully if the path does not exist.
pub fn try_resolve_path<'tcx>(
    tcx: TyCtxt<'tcx>,
    path: &[&str],
    namespace: Namespace,
) -> Option<ty::Instance<'tcx>> {
    let did = try_resolve_did(tcx, path, Some(namespace))?;
    Some(ty::Instance::mono(tcx, did))
}

/// Gets an instance for a path.
#[track_caller]
pub fn resolve_path<'tcx>(
    tcx: TyCtxt<'tcx>,
    path: &[&str],
    namespace: Namespace,
) -> ty::Instance<'tcx> {
    try_resolve_path(tcx, path, namespace)
        .unwrap_or_else(|| panic!("failed to find required Rust item: {path:?}"))
}

/// Gets the layout of a type at a path.
#[track_caller]
pub fn path_ty_layout<'tcx>(cx: &impl LayoutOf<'tcx>, path: &[&str]) -> TyAndLayout<'tcx> {
    let ty = resolve_path(cx.tcx(), path, Namespace::TypeNS).ty(cx.tcx(), cx.typing_env());
    cx.layout_of(ty).to_result().ok().unwrap()
}

/// Call `f` for each exported symbol.
pub fn iter_exported_symbols<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut f: impl FnMut(CrateNum, DefId) -> InterpResult<'tcx>,
) -> InterpResult<'tcx> {
    // First, the symbols in the local crate. We can't use `exported_symbols` here as that
    // skips `#[used]` statics (since `reachable_set` skips them in binary crates).
    // So we walk all HIR items ourselves instead.
    let crate_items = tcx.hir_crate_items(());
    for def_id in crate_items.definitions() {
        let exported = tcx.def_kind(def_id).has_codegen_attrs() && {
            let codegen_attrs = tcx.codegen_fn_attrs(def_id);
            codegen_attrs.contains_extern_indicator()
                || codegen_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)
                || codegen_attrs.flags.contains(CodegenFnAttrFlags::USED_COMPILER)
                || codegen_attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER)
        };
        if exported {
            f(LOCAL_CRATE, def_id.into())?;
        }
    }

    // Next, all our dependencies.
    // `dependency_formats` includes all the transitive informations needed to link a crate,
    // which is what we need here since we need to dig out `exported_symbols` from all transitive
    // dependencies.
    let dependency_formats = tcx.dependency_formats(());
    // Find the dependencies of the executable we are running.
    let dependency_format = dependency_formats
        .get(&CrateType::Executable)
        .expect("interpreting a non-executable crate");
    for cnum in dependency_format
        .iter_enumerated()
        .filter_map(|(num, &linkage)| (linkage != Linkage::NotLinked).then_some(num))
    {
        if cnum == LOCAL_CRATE {
            continue; // Already handled above
        }

        // We can ignore `_export_info` here: we are a Rust crate, and everything is exported
        // from a Rust crate.
        for &(symbol, _export_info) in tcx.exported_non_generic_symbols(cnum) {
            if let ExportedSymbol::NonGeneric(def_id) = symbol {
                f(cnum, def_id)?;
            }
        }
    }
    interp_ok(())
}

/// Convert a softfloat type to its corresponding hostfloat type.
pub trait ToHost {
    type HostFloat;
    fn to_host(self) -> Self::HostFloat;
}

/// Convert a hostfloat type to its corresponding softfloat type.
pub trait ToSoft {
    type SoftFloat;
    fn to_soft(self) -> Self::SoftFloat;
}

impl ToHost for rustc_apfloat::ieee::Double {
    type HostFloat = f64;

    fn to_host(self) -> Self::HostFloat {
        f64::from_bits(self.to_bits().try_into().unwrap())
    }
}

impl ToSoft for f64 {
    type SoftFloat = rustc_apfloat::ieee::Double;

    fn to_soft(self) -> Self::SoftFloat {
        Float::from_bits(self.to_bits().into())
    }
}

impl ToHost for rustc_apfloat::ieee::Single {
    type HostFloat = f32;

    fn to_host(self) -> Self::HostFloat {
        f32::from_bits(self.to_bits().try_into().unwrap())
    }
}

impl ToSoft for f32 {
    type SoftFloat = rustc_apfloat::ieee::Single;

    fn to_soft(self) -> Self::SoftFloat {
        Float::from_bits(self.to_bits().into())
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Checks if the given crate/module exists.
    fn have_module(&self, path: &[&str]) -> bool {
        try_resolve_did(*self.eval_context_ref().tcx, path, None).is_some()
    }

    /// Evaluates the scalar at the specified path.
    fn eval_path(&self, path: &[&str]) -> MPlaceTy<'tcx> {
        let this = self.eval_context_ref();
        let instance = resolve_path(*this.tcx, path, Namespace::ValueNS);
        // We don't give a span -- this isn't actually used directly by the program anyway.
        this.eval_global(instance).unwrap_or_else(|err| {
            panic!("failed to evaluate required Rust item: {path:?}\n{err:?}")
        })
    }
    fn eval_path_scalar(&self, path: &[&str]) -> Scalar {
        let this = self.eval_context_ref();
        let val = this.eval_path(path);
        this.read_scalar(&val)
            .unwrap_or_else(|err| panic!("failed to read required Rust item: {path:?}\n{err:?}"))
    }

    /// Helper function to get a `libc` constant as a `Scalar`.
    fn eval_libc(&self, name: &str) -> Scalar {
        if self.eval_context_ref().tcx.sess.target.os == "windows" {
            panic!(
                "`libc` crate is not reliably available on Windows targets; Miri should not use it there"
            );
        }
        self.eval_path_scalar(&["libc", name])
    }

    /// Helper function to get a `libc` constant as an `i32`.
    fn eval_libc_i32(&self, name: &str) -> i32 {
        // TODO: Cache the result.
        self.eval_libc(name).to_i32().unwrap_or_else(|_err| {
            panic!("required libc item has unexpected type (not `i32`): {name}")
        })
    }

    /// Helper function to get a `libc` constant as an `u32`.
    fn eval_libc_u32(&self, name: &str) -> u32 {
        // TODO: Cache the result.
        self.eval_libc(name).to_u32().unwrap_or_else(|_err| {
            panic!("required libc item has unexpected type (not `u32`): {name}")
        })
    }

    /// Helper function to get a `libc` constant as an `u64`.
    fn eval_libc_u64(&self, name: &str) -> u64 {
        // TODO: Cache the result.
        self.eval_libc(name).to_u64().unwrap_or_else(|_err| {
            panic!("required libc item has unexpected type (not `u64`): {name}")
        })
    }

    /// Helper function to get a `windows` constant as a `Scalar`.
    fn eval_windows(&self, module: &str, name: &str) -> Scalar {
        self.eval_context_ref().eval_path_scalar(&["std", "sys", "pal", "windows", module, name])
    }

    /// Helper function to get a `windows` constant as a `u32`.
    fn eval_windows_u32(&self, module: &str, name: &str) -> u32 {
        // TODO: Cache the result.
        self.eval_windows(module, name).to_u32().unwrap_or_else(|_err| {
            panic!("required Windows item has unexpected type (not `u32`): {module}::{name}")
        })
    }

    /// Helper function to get a `windows` constant as a `u64`.
    fn eval_windows_u64(&self, module: &str, name: &str) -> u64 {
        // TODO: Cache the result.
        self.eval_windows(module, name).to_u64().unwrap_or_else(|_err| {
            panic!("required Windows item has unexpected type (not `u64`): {module}::{name}")
        })
    }

    /// Helper function to get the `TyAndLayout` of a `libc` type
    fn libc_ty_layout(&self, name: &str) -> TyAndLayout<'tcx> {
        let this = self.eval_context_ref();
        if this.tcx.sess.target.os == "windows" {
            panic!(
                "`libc` crate is not reliably available on Windows targets; Miri should not use it there"
            );
        }
        path_ty_layout(this, &["libc", name])
    }

    /// Helper function to get the `TyAndLayout` of a `windows` type
    fn windows_ty_layout(&self, name: &str) -> TyAndLayout<'tcx> {
        let this = self.eval_context_ref();
        path_ty_layout(this, &["std", "sys", "pal", "windows", "c", name])
    }

    /// Helper function to get `TyAndLayout` of an array that consists of `libc` type.
    fn libc_array_ty_layout(&self, name: &str, size: u64) -> TyAndLayout<'tcx> {
        let this = self.eval_context_ref();
        let elem_ty_layout = this.libc_ty_layout(name);
        let array_ty = Ty::new_array(*this.tcx, elem_ty_layout.ty, size);
        this.layout_of(array_ty).unwrap()
    }

    /// Project to the given *named* field (which must be a struct or union type).
    fn try_project_field_named<P: Projectable<'tcx, Provenance>>(
        &self,
        base: &P,
        name: &str,
    ) -> InterpResult<'tcx, Option<P>> {
        let this = self.eval_context_ref();
        let adt = base.layout().ty.ty_adt_def().unwrap();
        for (idx, field) in adt.non_enum_variant().fields.iter_enumerated() {
            if field.name.as_str() == name {
                return interp_ok(Some(this.project_field(base, idx)?));
            }
        }
        interp_ok(None)
    }

    /// Project to the given *named* field (which must be a struct or union type).
    fn project_field_named<P: Projectable<'tcx, Provenance>>(
        &self,
        base: &P,
        name: &str,
    ) -> InterpResult<'tcx, P> {
        interp_ok(
            self.try_project_field_named(base, name)?
                .unwrap_or_else(|| bug!("no field named {} in type {}", name, base.layout().ty)),
        )
    }

    /// Write an int of the appropriate size to `dest`. The target type may be signed or unsigned,
    /// we try to do the right thing anyway. `i128` can fit all integer types except for `u128` so
    /// this method is fine for almost all integer types.
    fn write_int(
        &mut self,
        i: impl Into<i128>,
        dest: &impl Writeable<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        assert!(
            dest.layout().backend_repr.is_scalar(),
            "write_int on non-scalar type {}",
            dest.layout().ty
        );
        let val = if dest.layout().backend_repr.is_signed() {
            Scalar::from_int(i, dest.layout().size)
        } else {
            // `unwrap` can only fail here if `i` is negative
            Scalar::from_uint(u128::try_from(i.into()).unwrap(), dest.layout().size)
        };
        self.eval_context_mut().write_scalar(val, dest)
    }

    /// Write the first N fields of the given place.
    fn write_int_fields(
        &mut self,
        values: &[i128],
        dest: &impl Writeable<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        for (idx, &val) in values.iter().enumerate() {
            let idx = FieldIdx::from_usize(idx);
            let field = this.project_field(dest, idx)?;
            this.write_int(val, &field)?;
        }
        interp_ok(())
    }

    /// Write the given fields of the given place.
    fn write_int_fields_named(
        &mut self,
        values: &[(&str, i128)],
        dest: &impl Writeable<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        for &(name, val) in values.iter() {
            let field = this.project_field_named(dest, name)?;
            this.write_int(val, &field)?;
        }
        interp_ok(())
    }

    /// Write a 0 of the appropriate size to `dest`.
    fn write_null(&mut self, dest: &impl Writeable<'tcx, Provenance>) -> InterpResult<'tcx> {
        self.write_int(0, dest)
    }

    /// Test if this pointer equals 0.
    fn ptr_is_null(&self, ptr: Pointer) -> InterpResult<'tcx, bool> {
        interp_ok(ptr.addr().bytes() == 0)
    }

    /// Generate some random bytes, and write them to `dest`.
    fn gen_random(&mut self, ptr: Pointer, len: u64) -> InterpResult<'tcx> {
        // Some programs pass in a null pointer and a length of 0
        // to their platform's random-generation function (e.g. getrandom())
        // on Linux. For compatibility with these programs, we don't perform
        // any additional checks - it's okay if the pointer is invalid,
        // since we wouldn't actually be writing to it.
        if len == 0 {
            return interp_ok(());
        }
        let this = self.eval_context_mut();

        let mut data = vec![0; usize::try_from(len).unwrap()];

        if this.machine.communicate() {
            // Fill the buffer using the host's rng.
            getrandom::fill(&mut data)
                .map_err(|err| err_unsup_format!("host getrandom failed: {}", err))?;
        } else {
            let rng = this.machine.rng.get_mut();
            rng.fill_bytes(&mut data);
        }

        this.write_bytes_ptr(ptr, data.iter().copied())
    }

    /// Call a function: Push the stack frame and pass the arguments.
    /// For now, arguments must be scalars (so that the caller does not have to know the layout).
    ///
    /// If you do not provide a return place, a dangling zero-sized place will be created
    /// for your convenience.
    fn call_function(
        &mut self,
        f: ty::Instance<'tcx>,
        caller_abi: ExternAbi,
        args: &[ImmTy<'tcx>],
        dest: Option<&MPlaceTy<'tcx>>,
        stack_pop: StackPopCleanup,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // Get MIR.
        let mir = this.load_mir(f.def, None)?;
        let dest = match dest {
            Some(dest) => dest.clone(),
            None => MPlaceTy::fake_alloc_zst(this.layout_of(mir.return_ty())?),
        };

        // Construct a function pointer type representing the caller perspective.
        let sig = this.tcx.mk_fn_sig(
            args.iter().map(|a| a.layout.ty),
            dest.layout.ty,
            /*c_variadic*/ false,
            Safety::Safe,
            caller_abi,
        );
        let caller_fn_abi = this.fn_abi_of_fn_ptr(ty::Binder::dummy(sig), ty::List::empty())?;

        this.init_stack_frame(
            f,
            mir,
            caller_fn_abi,
            &args.iter().map(|a| FnArg::Copy(a.clone().into())).collect::<Vec<_>>(),
            /*with_caller_location*/ false,
            &dest.into(),
            stack_pop,
        )
    }

    /// Visits the memory covered by `place`, sensitive to freezing: the 2nd parameter
    /// of `action` will be true if this is frozen, false if this is in an `UnsafeCell`.
    /// The range is relative to `place`.
    fn visit_freeze_sensitive(
        &self,
        place: &MPlaceTy<'tcx>,
        size: Size,
        mut action: impl FnMut(AllocRange, bool) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        trace!("visit_frozen(place={:?}, size={:?})", *place, size);
        debug_assert_eq!(
            size,
            this.size_and_align_of_val(place)?
                .map(|(size, _)| size)
                .unwrap_or_else(|| place.layout.size)
        );
        // Store how far we proceeded into the place so far. Everything to the left of
        // this offset has already been handled, in the sense that the frozen parts
        // have had `action` called on them.
        let start_addr = place.ptr().addr();
        let mut cur_addr = start_addr;
        // Called when we detected an `UnsafeCell` at the given offset and size.
        // Calls `action` and advances `cur_ptr`.
        let mut unsafe_cell_action = |unsafe_cell_ptr: &Pointer, unsafe_cell_size: Size| {
            // We assume that we are given the fields in increasing offset order,
            // and nothing else changes.
            let unsafe_cell_addr = unsafe_cell_ptr.addr();
            assert!(unsafe_cell_addr >= cur_addr);
            let frozen_size = unsafe_cell_addr - cur_addr;
            // Everything between the cur_ptr and this `UnsafeCell` is frozen.
            if frozen_size != Size::ZERO {
                action(alloc_range(cur_addr - start_addr, frozen_size), /*frozen*/ true)?;
            }
            cur_addr += frozen_size;
            // This `UnsafeCell` is NOT frozen.
            if unsafe_cell_size != Size::ZERO {
                action(
                    alloc_range(cur_addr - start_addr, unsafe_cell_size),
                    /*frozen*/ false,
                )?;
            }
            cur_addr += unsafe_cell_size;
            // Done
            interp_ok(())
        };
        // Run a visitor
        {
            let mut visitor = UnsafeCellVisitor {
                ecx: this,
                unsafe_cell_action: |place| {
                    trace!("unsafe_cell_action on {:?}", place.ptr());
                    // We need a size to go on.
                    let unsafe_cell_size = this
                        .size_and_align_of_val(place)?
                        .map(|(size, _)| size)
                        // for extern types, just cover what we can
                        .unwrap_or_else(|| place.layout.size);
                    // Now handle this `UnsafeCell`, unless it is empty.
                    if unsafe_cell_size != Size::ZERO {
                        unsafe_cell_action(&place.ptr(), unsafe_cell_size)
                    } else {
                        interp_ok(())
                    }
                },
            };
            visitor.visit_value(place)?;
        }
        // The part between the end_ptr and the end of the place is also frozen.
        // So pretend there is a 0-sized `UnsafeCell` at the end.
        unsafe_cell_action(&place.ptr().wrapping_offset(size, this), Size::ZERO)?;
        // Done!
        return interp_ok(());

        /// Visiting the memory covered by a `MemPlace`, being aware of
        /// whether we are inside an `UnsafeCell` or not.
        struct UnsafeCellVisitor<'ecx, 'tcx, F>
        where
            F: FnMut(&MPlaceTy<'tcx>) -> InterpResult<'tcx>,
        {
            ecx: &'ecx MiriInterpCx<'tcx>,
            unsafe_cell_action: F,
        }

        impl<'ecx, 'tcx, F> ValueVisitor<'tcx, MiriMachine<'tcx>> for UnsafeCellVisitor<'ecx, 'tcx, F>
        where
            F: FnMut(&MPlaceTy<'tcx>) -> InterpResult<'tcx>,
        {
            type V = MPlaceTy<'tcx>;

            #[inline(always)]
            fn ecx(&self) -> &MiriInterpCx<'tcx> {
                self.ecx
            }

            fn aggregate_field_iter(
                memory_index: &IndexVec<FieldIdx, u32>,
            ) -> impl Iterator<Item = FieldIdx> + 'static {
                let inverse_memory_index = memory_index.invert_bijective_mapping();
                inverse_memory_index.into_iter()
            }

            // Hook to detect `UnsafeCell`.
            fn visit_value(&mut self, v: &MPlaceTy<'tcx>) -> InterpResult<'tcx> {
                trace!("UnsafeCellVisitor: {:?} {:?}", *v, v.layout.ty);
                let is_unsafe_cell = match v.layout.ty.kind() {
                    ty::Adt(adt, _) =>
                        Some(adt.did()) == self.ecx.tcx.lang_items().unsafe_cell_type(),
                    _ => false,
                };
                if is_unsafe_cell {
                    // We do not have to recurse further, this is an `UnsafeCell`.
                    (self.unsafe_cell_action)(v)
                } else if self.ecx.type_is_freeze(v.layout.ty) {
                    // This is `Freeze`, there cannot be an `UnsafeCell`
                    interp_ok(())
                } else if matches!(v.layout.fields, FieldsShape::Union(..)) {
                    // A (non-frozen) union. We fall back to whatever the type says.
                    (self.unsafe_cell_action)(v)
                } else {
                    // We want to not actually read from memory for this visit. So, before
                    // walking this value, we have to make sure it is not a
                    // `Variants::Multiple`.
                    // FIXME: the current logic here is layout-dependent, so enums with
                    // multiple variants where all but 1 are uninhabited will be recursed into.
                    // Is that truly what we want?
                    match v.layout.variants {
                        Variants::Multiple { .. } => {
                            // A multi-variant enum, or coroutine, or so.
                            // Treat this like a union: without reading from memory,
                            // we cannot determine the variant we are in. Reading from
                            // memory would be subject to Stacked Borrows rules, leading
                            // to all sorts of "funny" recursion.
                            // We only end up here if the type is *not* freeze, so we just call the
                            // `UnsafeCell` action.
                            (self.unsafe_cell_action)(v)
                        }
                        Variants::Single { .. } | Variants::Empty => {
                            // Proceed further, try to find where exactly that `UnsafeCell`
                            // is hiding.
                            self.walk_value(v)
                        }
                    }
                }
            }

            fn visit_union(
                &mut self,
                _v: &MPlaceTy<'tcx>,
                _fields: NonZero<usize>,
            ) -> InterpResult<'tcx> {
                bug!("we should have already handled unions in `visit_value`")
            }
        }
    }

    /// Helper function used inside the shims of foreign functions to check that isolation is
    /// disabled. It returns an error using the `name` of the foreign function if this is not the
    /// case.
    fn check_no_isolation(&self, name: &str) -> InterpResult<'tcx> {
        if !self.eval_context_ref().machine.communicate() {
            self.reject_in_isolation(name, RejectOpWith::Abort)?;
        }
        interp_ok(())
    }

    /// Helper function used inside the shims of foreign functions which reject the op
    /// when isolation is enabled. It is used to print a warning/backtrace about the rejection.
    fn reject_in_isolation(&self, op_name: &str, reject_with: RejectOpWith) -> InterpResult<'tcx> {
        let this = self.eval_context_ref();
        match reject_with {
            RejectOpWith::Abort => isolation_abort_error(op_name),
            RejectOpWith::WarningWithoutBacktrace => {
                let mut emitted_warnings = this.machine.reject_in_isolation_warned.borrow_mut();
                if !emitted_warnings.contains(op_name) {
                    // First time we are seeing this.
                    emitted_warnings.insert(op_name.to_owned());
                    this.tcx
                        .dcx()
                        .warn(format!("{op_name} was made to return an error due to isolation"));
                }

                interp_ok(())
            }
            RejectOpWith::Warning => {
                this.emit_diagnostic(NonHaltingDiagnostic::RejectedIsolatedOp(op_name.to_string()));
                interp_ok(())
            }
            RejectOpWith::NoWarning => interp_ok(()), // no warning
        }
    }

    /// Helper function used inside the shims of foreign functions to assert that the target OS
    /// is `target_os`. It panics showing a message with the `name` of the foreign function
    /// if this is not the case.
    fn assert_target_os(&self, target_os: &str, name: &str) {
        assert_eq!(
            self.eval_context_ref().tcx.sess.target.os,
            target_os,
            "`{name}` is only available on the `{target_os}` target OS",
        )
    }

    /// Helper function used inside shims of foreign functions to check that the target OS
    /// is one of `target_oses`. It returns an error containing the `name` of the foreign function
    /// in a message if this is not the case.
    fn check_target_os(&self, target_oses: &[&str], name: Symbol) -> InterpResult<'tcx> {
        let target_os = self.eval_context_ref().tcx.sess.target.os.as_ref();
        if !target_oses.contains(&target_os) {
            throw_unsup_format!("`{name}` is not supported on {target_os}");
        }
        interp_ok(())
    }

    /// Helper function used inside the shims of foreign functions to assert that the target OS
    /// is part of the UNIX family. It panics showing a message with the `name` of the foreign function
    /// if this is not the case.
    fn assert_target_os_is_unix(&self, name: &str) {
        assert!(self.target_os_is_unix(), "`{name}` is only available for unix targets",);
    }

    fn target_os_is_unix(&self) -> bool {
        self.eval_context_ref().tcx.sess.target.families.iter().any(|f| f == "unix")
    }

    /// Dereference a pointer operand to a place using `layout` instead of the pointer's declared type
    fn deref_pointer_as(
        &self,
        op: &impl Projectable<'tcx, Provenance>,
        layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx>> {
        let this = self.eval_context_ref();
        let ptr = this.read_pointer(op)?;
        interp_ok(this.ptr_to_mplace(ptr, layout))
    }

    /// Calculates the MPlaceTy given the offset and layout of an access on an operand
    fn deref_pointer_and_offset(
        &self,
        op: &impl Projectable<'tcx, Provenance>,
        offset: u64,
        base_layout: TyAndLayout<'tcx>,
        value_layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx>> {
        let this = self.eval_context_ref();
        let op_place = this.deref_pointer_as(op, base_layout)?;
        let offset = Size::from_bytes(offset);

        // Ensure that the access is within bounds.
        assert!(base_layout.size >= offset + value_layout.size);
        let value_place = op_place.offset(offset, value_layout, this)?;
        interp_ok(value_place)
    }

    fn deref_pointer_and_read(
        &self,
        op: &impl Projectable<'tcx, Provenance>,
        offset: u64,
        base_layout: TyAndLayout<'tcx>,
        value_layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_ref();
        let value_place = this.deref_pointer_and_offset(op, offset, base_layout, value_layout)?;
        this.read_scalar(&value_place)
    }

    fn deref_pointer_and_write(
        &mut self,
        op: &impl Projectable<'tcx, Provenance>,
        offset: u64,
        value: impl Into<Scalar>,
        base_layout: TyAndLayout<'tcx>,
        value_layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        let value_place = this.deref_pointer_and_offset(op, offset, base_layout, value_layout)?;
        this.write_scalar(value, &value_place)
    }

    /// Parse a `timespec` struct and return it as a `std::time::Duration`. It returns `None`
    /// if the value in the `timespec` struct is invalid. Some libc functions will return
    /// `EINVAL` in this case.
    fn read_timespec(&mut self, tp: &MPlaceTy<'tcx>) -> InterpResult<'tcx, Option<Duration>> {
        let this = self.eval_context_mut();
        let seconds_place = this.project_field(tp, FieldIdx::ZERO)?;
        let seconds_scalar = this.read_scalar(&seconds_place)?;
        let seconds = seconds_scalar.to_target_isize(this)?;
        let nanoseconds_place = this.project_field(tp, FieldIdx::ONE)?;
        let nanoseconds_scalar = this.read_scalar(&nanoseconds_place)?;
        let nanoseconds = nanoseconds_scalar.to_target_isize(this)?;

        interp_ok(
            try {
                // tv_sec must be non-negative.
                let seconds: u64 = seconds.try_into().ok()?;
                // tv_nsec must be non-negative.
                let nanoseconds: u32 = nanoseconds.try_into().ok()?;
                if nanoseconds >= 1_000_000_000 {
                    // tv_nsec must not be greater than 999,999,999.
                    None?
                }
                Duration::new(seconds, nanoseconds)
            },
        )
    }

    /// Read bytes from a byte slice.
    fn read_byte_slice<'a>(&'a self, slice: &ImmTy<'tcx>) -> InterpResult<'tcx, &'a [u8]>
    where
        'tcx: 'a,
    {
        let this = self.eval_context_ref();
        let (ptr, len) = slice.to_scalar_pair();
        let ptr = ptr.to_pointer(this)?;
        let len = len.to_target_usize(this)?;
        let bytes = this.read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(len))?;
        interp_ok(bytes)
    }

    /// Read a sequence of bytes until the first null terminator.
    fn read_c_str<'a>(&'a self, ptr: Pointer) -> InterpResult<'tcx, &'a [u8]>
    where
        'tcx: 'a,
    {
        let this = self.eval_context_ref();
        let size1 = Size::from_bytes(1);

        // Step 1: determine the length.
        let mut len = Size::ZERO;
        loop {
            // FIXME: We are re-getting the allocation each time around the loop.
            // Would be nice if we could somehow "extend" an existing AllocRange.
            let alloc = this.get_ptr_alloc(ptr.wrapping_offset(len, this), size1)?.unwrap(); // not a ZST, so we will get a result
            let byte = alloc.read_integer(alloc_range(Size::ZERO, size1))?.to_u8()?;
            if byte == 0 {
                break;
            } else {
                len += size1;
            }
        }

        // Step 2: get the bytes.
        this.read_bytes_ptr_strip_provenance(ptr, len)
    }

    /// Helper function to write a sequence of bytes with an added null-terminator, which is what
    /// the Unix APIs usually handle. This function returns `Ok((false, length))` without trying
    /// to write if `size` is not large enough to fit the contents of `c_str` plus a null
    /// terminator. It returns `Ok((true, length))` if the writing process was successful. The
    /// string length returned does include the null terminator.
    fn write_c_str(
        &mut self,
        c_str: &[u8],
        ptr: Pointer,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        // If `size` is smaller or equal than `bytes.len()`, writing `bytes` plus the required null
        // terminator to memory using the `ptr` pointer would cause an out-of-bounds access.
        let string_length = u64::try_from(c_str.len()).unwrap();
        let string_length = string_length.strict_add(1);
        if size < string_length {
            return interp_ok((false, string_length));
        }
        self.eval_context_mut()
            .write_bytes_ptr(ptr, c_str.iter().copied().chain(iter::once(0u8)))?;
        interp_ok((true, string_length))
    }

    /// Helper function to read a sequence of unsigned integers of the given size and alignment
    /// until the first null terminator.
    fn read_c_str_with_char_size<T>(
        &self,
        mut ptr: Pointer,
        size: Size,
        align: Align,
    ) -> InterpResult<'tcx, Vec<T>>
    where
        T: TryFrom<u128>,
        <T as TryFrom<u128>>::Error: std::fmt::Debug,
    {
        assert_ne!(size, Size::ZERO);

        let this = self.eval_context_ref();

        this.check_ptr_align(ptr, align)?;

        let mut wchars = Vec::new();
        loop {
            // FIXME: We are re-getting the allocation each time around the loop.
            // Would be nice if we could somehow "extend" an existing AllocRange.
            let alloc = this.get_ptr_alloc(ptr, size)?.unwrap(); // not a ZST, so we will get a result
            let wchar_int = alloc.read_integer(alloc_range(Size::ZERO, size))?.to_bits(size)?;
            if wchar_int == 0 {
                break;
            } else {
                wchars.push(wchar_int.try_into().unwrap());
                ptr = ptr.wrapping_offset(size, this);
            }
        }

        interp_ok(wchars)
    }

    /// Read a sequence of u16 until the first null terminator.
    fn read_wide_str(&self, ptr: Pointer) -> InterpResult<'tcx, Vec<u16>> {
        self.read_c_str_with_char_size(ptr, Size::from_bytes(2), Align::from_bytes(2).unwrap())
    }

    /// Helper function to write a sequence of u16 with an added 0x0000-terminator, which is what
    /// the Windows APIs usually handle. This function returns `Ok((false, length))` without trying
    /// to write if `size` is not large enough to fit the contents of `os_string` plus a null
    /// terminator. It returns `Ok((true, length))` if the writing process was successful. The
    /// string length returned does include the null terminator. Length is measured in units of
    /// `u16.`
    fn write_wide_str(
        &mut self,
        wide_str: &[u16],
        ptr: Pointer,
        size: u64,
    ) -> InterpResult<'tcx, (bool, u64)> {
        // If `size` is smaller or equal than `bytes.len()`, writing `bytes` plus the required
        // 0x0000 terminator to memory would cause an out-of-bounds access.
        let string_length = u64::try_from(wide_str.len()).unwrap();
        let string_length = string_length.strict_add(1);
        if size < string_length {
            return interp_ok((false, string_length));
        }

        // Store the UTF-16 string.
        let size2 = Size::from_bytes(2);
        let this = self.eval_context_mut();
        this.check_ptr_align(ptr, Align::from_bytes(2).unwrap())?;
        let mut alloc = this.get_ptr_alloc_mut(ptr, size2 * string_length)?.unwrap(); // not a ZST, so we will get a result
        for (offset, wchar) in wide_str.iter().copied().chain(iter::once(0x0000)).enumerate() {
            let offset = u64::try_from(offset).unwrap();
            alloc.write_scalar(alloc_range(size2 * offset, size2), Scalar::from_u16(wchar))?;
        }
        interp_ok((true, string_length))
    }

    /// Read a sequence of wchar_t until the first null terminator.
    /// Always returns a `Vec<u32>` no matter the size of `wchar_t`.
    fn read_wchar_t_str(&self, ptr: Pointer) -> InterpResult<'tcx, Vec<u32>> {
        let this = self.eval_context_ref();
        let wchar_t = if this.tcx.sess.target.os == "windows" {
            // We don't have libc on Windows so we have to hard-code the type ourselves.
            this.machine.layouts.u16
        } else {
            this.libc_ty_layout("wchar_t")
        };
        self.read_c_str_with_char_size(ptr, wchar_t.size, wchar_t.align.abi)
    }

    /// Check that the calling convention is what we expect.
    fn check_callconv<'a>(
        &self,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        exp_abi: CanonAbi,
    ) -> InterpResult<'a, ()> {
        if fn_abi.conv != exp_abi {
            throw_ub_format!(
                r#"calling a function with calling convention "{exp_abi}" using caller calling convention "{}""#,
                fn_abi.conv
            );
        }
        interp_ok(())
    }

    fn frame_in_std(&self) -> bool {
        let this = self.eval_context_ref();
        let frame = this.frame();
        // Make an attempt to get at the instance of the function this is inlined from.
        let instance: Option<_> = try {
            let scope = frame.current_source_info()?.scope;
            let inlined_parent = frame.body().source_scopes[scope].inlined_parent_scope?;
            let source = &frame.body().source_scopes[inlined_parent];
            source.inlined.expect("inlined_parent_scope points to scope without inline info").0
        };
        // Fall back to the instance of the function itself.
        let instance = instance.unwrap_or(frame.instance());
        // Now check the crate it is in. We could try to be clever here and e.g. check if this is
        // the same crate as `start_fn`, but that would not work for running std tests in Miri, so
        // we'd need some more hacks anyway. So we just check the name of the crate. If someone
        // calls their crate `std` then we'll just let them keep the pieces.
        let frame_crate = this.tcx.def_path(instance.def_id()).krate;
        let crate_name = this.tcx.crate_name(frame_crate);
        let crate_name = crate_name.as_str();
        // On miri-test-libstd, the name of the crate is different.
        crate_name == "std" || crate_name == "std_miri_test"
    }

    fn check_abi_and_shim_symbol_clash(
        &mut self,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        exp_abi: CanonAbi,
        link_name: Symbol,
    ) -> InterpResult<'tcx, ()> {
        self.check_callconv(abi, exp_abi)?;
        if let Some((body, instance)) = self.eval_context_mut().lookup_exported_symbol(link_name)? {
            // If compiler-builtins is providing the symbol, then don't treat it as a clash.
            // We'll use our built-in implementation in `emulate_foreign_item_inner` for increased
            // performance. Note that this means we won't catch any undefined behavior in
            // compiler-builtins when running other crates, but Miri can still be run on
            // compiler-builtins itself (or any crate that uses it as a normal dependency)
            if self.eval_context_ref().tcx.is_compiler_builtins(instance.def_id().krate) {
                return interp_ok(());
            }

            throw_machine_stop!(TerminationInfo::SymbolShimClashing {
                link_name,
                span: body.span.data(),
            })
        }
        interp_ok(())
    }

    fn check_shim<'a, const N: usize>(
        &mut self,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        exp_abi: CanonAbi,
        link_name: Symbol,
        args: &'a [OpTy<'tcx>],
    ) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]> {
        self.check_abi_and_shim_symbol_clash(abi, exp_abi, link_name)?;

        if abi.c_variadic {
            throw_ub_format!(
                "calling a non-variadic function with a variadic caller-side signature"
            );
        }
        if let Ok(ops) = args.try_into() {
            return interp_ok(ops);
        }
        throw_ub_format!(
            "incorrect number of arguments for `{link_name}`: got {}, expected {}",
            args.len(),
            N
        )
    }

    /// Check that the given `caller_fn_abi` matches the expected ABI described by
    /// `callee_abi`, `callee_input_tys`, `callee_output_ty`, and then returns the list of
    /// arguments.
    fn check_shim_abi<'a, const N: usize>(
        &mut self,
        link_name: Symbol,
        caller_fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        callee_abi: ExternAbi,
        callee_input_tys: [Ty<'tcx>; N],
        callee_output_ty: Ty<'tcx>,
        caller_args: &'a [OpTy<'tcx>],
    ) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]> {
        let this = self.eval_context_mut();
        let mut inputs_and_output = callee_input_tys.to_vec();
        inputs_and_output.push(callee_output_ty);
        let fn_sig_binder = Binder::dummy(FnSig {
            inputs_and_output: this.machine.tcx.mk_type_list(&inputs_and_output),
            c_variadic: false,
            // This does not matter for the ABI.
            safety: Safety::Safe,
            abi: callee_abi,
        });
        let callee_fn_abi = this.fn_abi_of_fn_ptr(fn_sig_binder, Default::default())?;

        this.check_abi_and_shim_symbol_clash(caller_fn_abi, callee_fn_abi.conv, link_name)?;

        if caller_fn_abi.c_variadic {
            throw_ub_format!(
                "ABI mismatch: calling a non-variadic function with a variadic caller-side signature"
            );
        }

        if callee_fn_abi.fixed_count != caller_fn_abi.fixed_count {
            throw_ub_format!(
                "ABI mismatch: expected {} arguments, found {} arguments ",
                callee_fn_abi.fixed_count,
                caller_fn_abi.fixed_count
            );
        }

        if callee_fn_abi.can_unwind && !caller_fn_abi.can_unwind {
            throw_ub_format!(
                "ABI mismatch: callee may unwind, but caller-side signature prohibits unwinding",
            );
        }

        if !this.check_argument_compat(&caller_fn_abi.ret, &callee_fn_abi.ret)? {
            throw_ub!(AbiMismatchReturn {
                caller_ty: caller_fn_abi.ret.layout.ty,
                callee_ty: callee_fn_abi.ret.layout.ty
            });
        }

        if let Some(index) = caller_fn_abi
            .args
            .iter()
            .zip(callee_fn_abi.args.iter())
            .map(|(caller_arg, callee_arg)| this.check_argument_compat(caller_arg, callee_arg))
            .collect::<InterpResult<'tcx, Vec<bool>>>()?
            .into_iter()
            .position(|b| !b)
        {
            throw_ub!(AbiMismatchArgument {
                caller_ty: caller_fn_abi.args[index].layout.ty,
                callee_ty: callee_fn_abi.args[index].layout.ty
            });
        }

        if let Ok(ops) = caller_args.try_into() {
            return interp_ok(ops);
        }
        unreachable!()
    }

    /// Check shim for variadic function.
    /// Returns a tuple that consisting of an array of fixed args, and a slice of varargs.
    fn check_shim_variadic<'a, const N: usize>(
        &mut self,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        exp_abi: CanonAbi,
        link_name: Symbol,
        args: &'a [OpTy<'tcx>],
    ) -> InterpResult<'tcx, (&'a [OpTy<'tcx>; N], &'a [OpTy<'tcx>])>
    where
        &'a [OpTy<'tcx>; N]: TryFrom<&'a [OpTy<'tcx>]>,
    {
        self.check_abi_and_shim_symbol_clash(abi, exp_abi, link_name)?;

        if !abi.c_variadic {
            throw_ub_format!(
                "calling a variadic function with a non-variadic caller-side signature"
            );
        }
        if abi.fixed_count != u32::try_from(N).unwrap() {
            throw_ub_format!(
                "incorrect number of fixed arguments for variadic function `{}`: got {}, expected {N}",
                link_name.as_str(),
                abi.fixed_count
            )
        }
        if let Some(args) = args.split_first_chunk() {
            return interp_ok(args);
        }
        panic!("mismatch between signature and `args` slice");
    }

    /// Mark a machine allocation that was just created as immutable.
    fn mark_immutable(&mut self, mplace: &MPlaceTy<'tcx>) {
        let this = self.eval_context_mut();
        // This got just allocated, so there definitely is a pointer here.
        let provenance = mplace.ptr().into_pointer_or_addr().unwrap().provenance;
        this.alloc_mark_immutable(provenance.get_alloc_id().unwrap()).unwrap();
    }

    /// Converts `src` from floating point to integer type `dest_ty`
    /// after rounding with mode `round`.
    /// Returns `None` if `f` is NaN or out of range.
    fn float_to_int_checked(
        &self,
        src: &ImmTy<'tcx>,
        cast_to: TyAndLayout<'tcx>,
        round: rustc_apfloat::Round,
    ) -> InterpResult<'tcx, Option<ImmTy<'tcx>>> {
        let this = self.eval_context_ref();

        fn float_to_int_inner<'tcx, F: rustc_apfloat::Float>(
            ecx: &MiriInterpCx<'tcx>,
            src: F,
            cast_to: TyAndLayout<'tcx>,
            round: rustc_apfloat::Round,
        ) -> (Scalar, rustc_apfloat::Status) {
            let int_size = cast_to.layout.size;
            match cast_to.ty.kind() {
                // Unsigned
                ty::Uint(_) => {
                    let res = src.to_u128_r(int_size.bits_usize(), round, &mut false);
                    (Scalar::from_uint(res.value, int_size), res.status)
                }
                // Signed
                ty::Int(_) => {
                    let res = src.to_i128_r(int_size.bits_usize(), round, &mut false);
                    (Scalar::from_int(res.value, int_size), res.status)
                }
                // Nothing else
                _ =>
                    span_bug!(
                        ecx.cur_span(),
                        "attempted float-to-int conversion with non-int output type {}",
                        cast_to.ty,
                    ),
            }
        }

        let ty::Float(fty) = src.layout.ty.kind() else {
            bug!("float_to_int_checked: non-float input type {}", src.layout.ty)
        };

        let (val, status) = match fty {
            FloatTy::F16 =>
                float_to_int_inner::<Half>(this, src.to_scalar().to_f16()?, cast_to, round),
            FloatTy::F32 =>
                float_to_int_inner::<Single>(this, src.to_scalar().to_f32()?, cast_to, round),
            FloatTy::F64 =>
                float_to_int_inner::<Double>(this, src.to_scalar().to_f64()?, cast_to, round),
            FloatTy::F128 =>
                float_to_int_inner::<Quad>(this, src.to_scalar().to_f128()?, cast_to, round),
        };

        if status.intersects(
            rustc_apfloat::Status::INVALID_OP
                | rustc_apfloat::Status::OVERFLOW
                | rustc_apfloat::Status::UNDERFLOW,
        ) {
            // Floating point value is NaN (flagged with INVALID_OP) or outside the range
            // of values of the integer type (flagged with OVERFLOW or UNDERFLOW).
            interp_ok(None)
        } else {
            // Floating point value can be represented by the integer type after rounding.
            // The INEXACT flag is ignored on purpose to allow rounding.
            interp_ok(Some(ImmTy::from_scalar(val, cast_to)))
        }
    }

    /// Returns an integer type that is twice wide as `ty`
    fn get_twice_wide_int_ty(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let this = self.eval_context_ref();
        match ty.kind() {
            // Unsigned
            ty::Uint(UintTy::U8) => this.tcx.types.u16,
            ty::Uint(UintTy::U16) => this.tcx.types.u32,
            ty::Uint(UintTy::U32) => this.tcx.types.u64,
            ty::Uint(UintTy::U64) => this.tcx.types.u128,
            // Signed
            ty::Int(IntTy::I8) => this.tcx.types.i16,
            ty::Int(IntTy::I16) => this.tcx.types.i32,
            ty::Int(IntTy::I32) => this.tcx.types.i64,
            ty::Int(IntTy::I64) => this.tcx.types.i128,
            _ => span_bug!(this.cur_span(), "unexpected type: {ty:?}"),
        }
    }

    /// Checks that target feature `target_feature` is enabled.
    ///
    /// If not enabled, emits an UB error that states that the feature is
    /// required by `intrinsic`.
    fn expect_target_feature_for_intrinsic(
        &self,
        intrinsic: Symbol,
        target_feature: &str,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_ref();
        if !this.tcx.sess.unstable_target_features.contains(&Symbol::intern(target_feature)) {
            throw_ub_format!(
                "attempted to call intrinsic `{intrinsic}` that requires missing target feature {target_feature}"
            );
        }
        interp_ok(())
    }

    /// Lookup an array of immediates stored as a linker section of name `name`.
    fn lookup_link_section(&mut self, name: &str) -> InterpResult<'tcx, Vec<ImmTy<'tcx>>> {
        let this = self.eval_context_mut();
        let tcx = this.tcx.tcx;

        let mut array = vec![];

        iter_exported_symbols(tcx, |_cnum, def_id| {
            let attrs = tcx.codegen_fn_attrs(def_id);
            let Some(link_section) = attrs.link_section else {
                return interp_ok(());
            };
            if link_section.as_str() == name {
                let instance = ty::Instance::mono(tcx, def_id);
                let const_val = this.eval_global(instance).unwrap_or_else(|err| {
                    panic!(
                        "failed to evaluate static in required link_section: {def_id:?}\n{err:?}"
                    )
                });
                let val = this.read_immediate(&const_val)?;
                array.push(val);
            }
            interp_ok(())
        })?;

        interp_ok(array)
    }

    fn mangle_internal_symbol<'a>(&'a mut self, name: &'static str) -> &'a str
    where
        'tcx: 'a,
    {
        let this = self.eval_context_mut();
        let tcx = *this.tcx;
        this.machine
            .mangle_internal_symbol_cache
            .entry(name)
            .or_insert_with(|| mangle_internal_symbol(tcx, name))
    }
}

impl<'tcx> MiriMachine<'tcx> {
    /// Get the current span in the topmost function which is workspace-local and not
    /// `#[track_caller]`.
    /// This function is backed by a cache, and can be assumed to be very fast.
    /// It will work even when the stack is empty.
    pub fn current_span(&self) -> Span {
        self.threads.active_thread_ref().current_span()
    }

    /// Returns the span of the *caller* of the current operation, again
    /// walking down the stack to find the closest frame in a local crate, if the caller of the
    /// current operation is not in a local crate.
    /// This is useful when we are processing something which occurs on function-entry and we want
    /// to point at the call to the function, not the function definition generally.
    pub fn caller_span(&self) -> Span {
        // We need to go down at least to the caller (len - 2), or however
        // far we have to go to find a frame in a local crate which is also not #[track_caller].
        let frame_idx = self.top_user_relevant_frame().unwrap();
        let frame_idx = cmp::min(frame_idx, self.stack().len().saturating_sub(2));
        self.stack()[frame_idx].current_span()
    }

    fn stack(&self) -> &[Frame<'tcx, Provenance, machine::FrameExtra<'tcx>>] {
        self.threads.active_thread_stack()
    }

    fn top_user_relevant_frame(&self) -> Option<usize> {
        self.threads.active_thread_ref().top_user_relevant_frame()
    }

    /// This is the source of truth for the `is_user_relevant` flag in our `FrameExtra`.
    pub fn is_user_relevant(&self, frame: &Frame<'tcx, Provenance>) -> bool {
        let def_id = frame.instance().def_id();
        (def_id.is_local() || self.local_crates.contains(&def_id.krate))
            && !frame.instance().def.requires_caller_location(self.tcx)
    }
}

/// Check that the number of args is what we expect.
pub fn check_intrinsic_arg_count<'a, 'tcx, const N: usize>(
    args: &'a [OpTy<'tcx>],
) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]>
where
    &'a [OpTy<'tcx>; N]: TryFrom<&'a [OpTy<'tcx>]>,
{
    if let Ok(ops) = args.try_into() {
        return interp_ok(ops);
    }
    throw_ub_format!(
        "incorrect number of arguments for intrinsic: got {}, expected {}",
        args.len(),
        N
    )
}

/// Check that the number of varargs is at least the minimum what we expect.
/// Fixed args should not be included.
/// Use `check_vararg_fixed_arg_count` to extract the varargs slice from full function arguments.
pub fn check_min_vararg_count<'a, 'tcx, const N: usize>(
    name: &'a str,
    args: &'a [OpTy<'tcx>],
) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]> {
    if let Some((ops, _)) = args.split_first_chunk() {
        return interp_ok(ops);
    }
    throw_ub_format!(
        "not enough variadic arguments for `{name}`: got {}, expected at least {}",
        args.len(),
        N
    )
}

pub fn isolation_abort_error<'tcx>(name: &str) -> InterpResult<'tcx> {
    throw_machine_stop!(TerminationInfo::UnsupportedInIsolation(format!(
        "{name} not available when isolation is enabled",
    )))
}

/// Retrieve the list of local crates that should have been passed by cargo-miri in
/// MIRI_LOCAL_CRATES and turn them into `CrateNum`s.
pub fn get_local_crates(tcx: TyCtxt<'_>) -> Vec<CrateNum> {
    // Convert the local crate names from the passed-in config into CrateNums so that they can
    // be looked up quickly during execution
    let local_crate_names = std::env::var("MIRI_LOCAL_CRATES")
        .map(|crates| crates.split(',').map(|krate| krate.to_string()).collect::<Vec<_>>())
        .unwrap_or_default();
    let mut local_crates = Vec::new();
    for &crate_num in tcx.crates(()) {
        let name = tcx.crate_name(crate_num);
        let name = name.as_str();
        if local_crate_names.iter().any(|local_name| local_name == name) {
            local_crates.push(crate_num);
        }
    }
    local_crates
}

pub(crate) fn bool_to_simd_element(b: bool, size: Size) -> Scalar {
    // SIMD uses all-1 as pattern for "true". In two's complement,
    // -1 has all its bits set to one and `from_int` will truncate or
    // sign-extend it to `size` as required.
    let val = if b { -1 } else { 0 };
    Scalar::from_int(val, size)
}

pub(crate) fn simd_element_to_bool(elem: ImmTy<'_>) -> InterpResult<'_, bool> {
    assert!(
        matches!(elem.layout.ty.kind(), ty::Int(_) | ty::Uint(_)),
        "SIMD mask element type must be an integer, but this is `{}`",
        elem.layout.ty
    );
    let val = elem.to_scalar().to_int(elem.layout.size)?;
    interp_ok(match val {
        0 => false,
        -1 => true,
        _ => throw_ub_format!("each element of a SIMD mask must be all-0-bits or all-1-bits"),
    })
}

/// Check whether an operation that writes to a target buffer was successful.
/// Accordingly select return value.
/// Local helper function to be used in Windows shims.
pub(crate) fn windows_check_buffer_size((success, len): (bool, u64)) -> u32 {
    if success {
        // If the function succeeds, the return value is the number of characters stored in the target buffer,
        // not including the terminating null character.
        u32::try_from(len.strict_sub(1)).unwrap()
    } else {
        // If the target buffer was not large enough to hold the data, the return value is the buffer size, in characters,
        // required to hold the string and its terminating null character.
        u32::try_from(len).unwrap()
    }
}

/// We don't support 16-bit systems, so let's have ergonomic conversion from `u32` to `usize`.
pub trait ToUsize {
    fn to_usize(self) -> usize;
}

impl ToUsize for u32 {
    fn to_usize(self) -> usize {
        self.try_into().unwrap()
    }
}

/// Similarly, a maximum address size of `u64` is assumed widely here, so let's have ergonomic
/// converion from `usize` to `u64`.
pub trait ToU64 {
    fn to_u64(self) -> u64;
}

impl ToU64 for usize {
    fn to_u64(self) -> u64 {
        self.try_into().unwrap()
    }
}
