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

        let llty = bx.fn_ptr_backend_type(fn_abi);
        let ptr_size = bx.data_layout().pointer_size;
        let vtable_byte_offset = self.0 * ptr_size.bytes();

        load_vtable(bx, llvtable, llty, vtable_byte_offset, ty, nonnull)
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

        let llty = bx.type_isize();
        let ptr_size = bx.data_layout().pointer_size;
        let vtable_byte_offset = self.0 * ptr_size.bytes();

        load_vtable(bx, llvtable, llty, vtable_byte_offset, ty, false)
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
            && let ty::Dynamic(data, _, _) = ty.kind()
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
    let vtable_const = cx.const_data_from_alloc(vtable_allocation);
    let align = cx.data_layout().pointer_align.abi;
    let vtable = cx.static_addr_of(vtable_const, align, Some("vtable"));

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
) -> Bx::Value {
    let ptr_align = bx.data_layout().pointer_align.abi;

    if bx.cx().sess().opts.unstable_opts.virtual_function_elimination
        && bx.cx().sess().lto() == Lto::Fat
    {
        if let Some(trait_ref) = dyn_trait_in_self(bx.tcx(), ty) {
            let typeid = bx.typeid_metadata(typeid_for_trait_ref(bx.tcx(), trait_ref)).unwrap();
            let func = bx.type_checked_load(llvtable, vtable_byte_offset, typeid);
            return func;
        } else if nonnull {
            bug!("load nonnull value from a vtable without a principal trait")
        }
    }

    let gep = bx.inbounds_ptradd(llvtable, bx.const_usize(vtable_byte_offset));
    let ptr = bx.load(llty, gep, ptr_align);
    // VTable loads are invariant.
    bx.set_invariant_load(ptr);
    if nonnull {
        bx.nonnull_metadata(ptr);
    }
    ptr
}
