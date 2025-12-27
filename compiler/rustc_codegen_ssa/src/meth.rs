use rustc_middle::bug;
use rustc_middle::ty::{self, GenericArgKind, Ty, TyCtxt};
use rustc_session::config::Lto;
use rustc_symbol_mangling::typeid_for_trait_ref;
use rustc_target::callconv::FnAbi;
use tracing::{debug, instrument};

use crate::traits::*;

#[derive(Copy, Clone, Debug)]
pub(crate) struct VirtualIndex(u64);

impl<'a, 'tcx> VirtualIndex {
    pub(crate) fn from_index(index: usize) -> Self {
        VirtualIndex(index as u64)
    }

    fn get_fn_inner<Bx: BuilderMethods<'a, 'tcx>>(
        self,
        bx: &mut Bx,
        llvtable: Bx::Value,
        ty: Ty<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        nonnull: bool,
    ) -> Bx::Value {
        // Load the function pointer from the object.
        debug!("get_fn({llvtable:?}, {ty:?}, {self:?})");
        let ptr_size = bx.data_layout().pointer_size();
        let vtable_byte_offset = self.0 * ptr_size.bytes();

        if bx.cx().sess().opts.unstable_opts.experimental_relative_rust_abi_vtables {
            let llty = bx.vtable_component_type(fn_abi);
            load_vtable(
                bx,
                llvtable,
                llty,
                vtable_byte_offset / 2,
                ty,
                nonnull,
                /*load_relative*/ true,
            )
        } else {
            let llty = bx.fn_ptr_backend_type(fn_abi);
            load_vtable(
                bx,
                llvtable,
                llty,
                vtable_byte_offset,
                ty,
                nonnull,
                /*load_relative*/ false,
            )
        }
    }

    pub(crate) fn get_optional_fn<Bx: BuilderMethods<'a, 'tcx>>(
        self,
        bx: &mut Bx,
        llvtable: Bx::Value,
        ty: Ty<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    ) -> Bx::Value {
        self.get_fn_inner(bx, llvtable, ty, fn_abi, false)
    }

    pub(crate) fn get_fn<Bx: BuilderMethods<'a, 'tcx>>(
        self,
        bx: &mut Bx,
        llvtable: Bx::Value,
        ty: Ty<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    ) -> Bx::Value {
        self.get_fn_inner(bx, llvtable, ty, fn_abi, true)
    }

    pub(crate) fn get_usize<Bx: BuilderMethods<'a, 'tcx>>(
        self,
        bx: &mut Bx,
        llvtable: Bx::Value,
        ty: Ty<'tcx>,
    ) -> Bx::Value {
        // Load the data pointer from the object.
        debug!("get_int({:?}, {:?})", llvtable, self);
        let ptr_size = bx.data_layout().pointer_size();
        let vtable_byte_offset = self.0 * ptr_size.bytes();

        if bx.cx().sess().opts.unstable_opts.experimental_relative_rust_abi_vtables {
            let llty = bx.type_i32();
            let val = load_vtable(
                bx,
                llvtable,
                llty,
                vtable_byte_offset / 2,
                ty,
                false,
                /*load_relative*/ false,
            );
            bx.zext(val, bx.type_isize())
        } else {
            let llty = bx.type_isize();
            load_vtable(
                bx,
                llvtable,
                llty,
                vtable_byte_offset,
                ty,
                false,
                /*load_relative*/ false,
            )
        }
    }
}

/// This takes a valid `self` receiver type and extracts the principal trait
/// ref of the type. Return `None` if there is no principal trait.
fn dyn_trait_in_self<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Option<ty::ExistentialTraitRef<'tcx>> {
    for arg in ty.peel_refs().walk() {
        if let GenericArgKind::Type(ty) = arg.kind()
            && let ty::Dynamic(data, _) = ty.kind()
        {
            // FIXME(arbitrary_self_types): This is likely broken for receivers which
            // have a "non-self" trait objects as a generic argument.
            return data
                .principal()
                .map(|principal| tcx.instantiate_bound_regions_with_erased(principal));
        }
    }

    bug!("expected a `dyn Trait` ty, found {ty:?}")
}

/// Creates a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
///
/// The vtables are cached instead of created on every call.
///
/// The `trait_ref` encodes the erased self type. Hence if we are
/// making an object `Foo<dyn Trait>` from a value of type `Foo<T>`, then
/// `trait_ref` would map `T: Trait`.
#[instrument(level = "debug", skip(cx))]
pub(crate) fn get_vtable<'tcx, Cx: CodegenMethods<'tcx>>(
    cx: &Cx,
    ty: Ty<'tcx>,
    trait_ref: Option<ty::ExistentialTraitRef<'tcx>>,
) -> Cx::Value {
    let tcx = cx.tcx();

    // Check the cache.
    if let Some(&val) = cx.vtables().borrow().get(&(ty, trait_ref)) {
        return val;
    }

    let vtable_alloc_id = tcx.vtable_allocation((ty, trait_ref));
    let vtable_allocation = tcx.global_alloc(vtable_alloc_id).unwrap_memory();
    let num_entries = {
        if let Some(trait_ref) = trait_ref {
            let trait_ref = trait_ref.with_self_ty(tcx, ty);
            let trait_ref = tcx.erase_and_anonymize_regions(trait_ref);
            tcx.vtable_entries(trait_ref)
        } else {
            TyCtxt::COMMON_VTABLE_ENTRIES
        }
    }
    .len();
    let vtable = cx.construct_vtable(vtable_allocation, num_entries as u64);

    cx.apply_vcall_visibility_metadata(ty, trait_ref, vtable);
    cx.create_vtable_debuginfo(ty, trait_ref, vtable);
    cx.vtables().borrow_mut().insert((ty, trait_ref), vtable);
    vtable
}

/// Call this function whenever you need to load a vtable.
pub(crate) fn load_vtable<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    llvtable: Bx::Value,
    llty: Bx::Type,
    vtable_byte_offset: u64,
    ty: Ty<'tcx>,
    nonnull: bool,
    load_relative: bool,
) -> Bx::Value {
    let ptr_align = bx.data_layout().pointer_align().abi;

    if bx.cx().sess().opts.unstable_opts.virtual_function_elimination
        && bx.cx().sess().lto() == Lto::Fat
    {
        if let Some(trait_ref) = dyn_trait_in_self(bx.tcx(), ty) {
            let typeid =
                bx.typeid_metadata(typeid_for_trait_ref(bx.tcx(), trait_ref).as_bytes()).unwrap();
            // FIXME: Add correct intrinsic for RV here.
            let func = bx.type_checked_load(llvtable, vtable_byte_offset, typeid);
            return func;
        } else if nonnull {
            bug!("load nonnull value from a vtable without a principal trait")
        }
    }

    let ptr = if load_relative {
        bx.load_relative(llvtable, bx.const_i32(vtable_byte_offset.try_into().unwrap()))
    } else {
        let gep = bx.inbounds_ptradd(llvtable, bx.const_usize(vtable_byte_offset));
        bx.load(llty, gep, ptr_align)
    };

    // VTable loads are invariant.
    bx.set_invariant_load(ptr);
    // FIXME: The verifier complains with
    //
    //   nonnull applies only to load instructions, use attributes for calls or invokes
    //   @llvm.load.relative.i32  (ptr nonnull %13, i32 12), !dbg !4323, !invariant.load !27, !noalias !27, !nonnull !27
    //
    // For now, do not mark the load relative intrinsic with nonnull, but I think it should be fine
    // to do so since it's effectively a load.
    if nonnull && !load_relative {
        bx.nonnull_metadata(ptr);
    }
    ptr
}
