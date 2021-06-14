use crate::traits::*;

use rustc_middle::ty::{self, Instance, Ty, VtblEntry, COMMON_VTABLE_ENTRIES};
use rustc_target::abi::call::FnAbi;

#[derive(Copy, Clone, Debug)]
pub struct VirtualIndex(u64);

impl<'a, 'tcx> VirtualIndex {
    pub fn from_index(index: usize) -> Self {
        VirtualIndex(index as u64)
    }

    pub fn get_fn<Bx: BuilderMethods<'a, 'tcx>>(
        self,
        bx: &mut Bx,
        llvtable: Bx::Value,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    ) -> Bx::Value {
        // Load the data pointer from the object.
        debug!("get_fn({:?}, {:?})", llvtable, self);

        let llvtable = bx.pointercast(llvtable, bx.type_ptr_to(bx.fn_ptr_backend_type(fn_abi)));
        let ptr_align = bx.tcx().data_layout.pointer_align.abi;
        let gep = bx.inbounds_gep(llvtable, &[bx.const_usize(self.0)]);
        let ptr = bx.load(gep, ptr_align);
        bx.nonnull_metadata(ptr);
        // Vtable loads are invariant.
        bx.set_invariant_load(ptr);
        ptr
    }

    pub fn get_usize<Bx: BuilderMethods<'a, 'tcx>>(
        self,
        bx: &mut Bx,
        llvtable: Bx::Value,
    ) -> Bx::Value {
        // Load the data pointer from the object.
        debug!("get_int({:?}, {:?})", llvtable, self);

        let llvtable = bx.pointercast(llvtable, bx.type_ptr_to(bx.type_isize()));
        let usize_align = bx.tcx().data_layout.pointer_align.abi;
        let gep = bx.inbounds_gep(llvtable, &[bx.const_usize(self.0)]);
        let ptr = bx.load(gep, usize_align);
        // Vtable loads are invariant.
        bx.set_invariant_load(ptr);
        ptr
    }
}

/// Creates a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
///
/// The vtables are cached instead of created on every call.
///
/// The `trait_ref` encodes the erased self type. Hence if we are
/// making an object `Foo<dyn Trait>` from a value of type `Foo<T>`, then
/// `trait_ref` would map `T: Trait`.
pub fn get_vtable<'tcx, Cx: CodegenMethods<'tcx>>(
    cx: &Cx,
    ty: Ty<'tcx>,
    trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
) -> Cx::Value {
    let tcx = cx.tcx();

    debug!("get_vtable(ty={:?}, trait_ref={:?})", ty, trait_ref);

    // Check the cache.
    if let Some(&val) = cx.vtables().borrow().get(&(ty, trait_ref)) {
        return val;
    }

    // Not in the cache; build it.
    let nullptr = cx.const_null(cx.type_i8p_ext(cx.data_layout().instruction_address_space));

    let vtable_entries = if let Some(trait_ref) = trait_ref {
        tcx.vtable_entries(trait_ref.with_self_ty(tcx, ty))
    } else {
        COMMON_VTABLE_ENTRIES
    };

    let layout = cx.layout_of(ty);
    // /////////////////////////////////////////////////////////////////////////////////////////////
    // If you touch this code, be sure to also make the corresponding changes to
    // `get_vtable` in `rust_mir/interpret/traits.rs`.
    // /////////////////////////////////////////////////////////////////////////////////////////////
    let components: Vec<_> = vtable_entries
        .iter()
        .map(|entry| match entry {
            VtblEntry::MetadataDropInPlace => {
                cx.get_fn_addr(Instance::resolve_drop_in_place(cx.tcx(), ty))
            }
            VtblEntry::MetadataSize => cx.const_usize(layout.size.bytes()),
            VtblEntry::MetadataAlign => cx.const_usize(layout.align.abi.bytes()),
            VtblEntry::Vacant => nullptr,
            VtblEntry::Method(def_id, substs) => cx.get_fn_addr(
                ty::Instance::resolve_for_vtable(
                    cx.tcx(),
                    ty::ParamEnv::reveal_all(),
                    *def_id,
                    substs,
                )
                .unwrap()
                .polymorphize(cx.tcx()),
            ),
        })
        .collect();

    let vtable_const = cx.const_struct(&components, false);
    let align = cx.data_layout().pointer_align.abi;
    let vtable = cx.static_addr_of(vtable_const, align, Some("vtable"));

    cx.create_vtable_metadata(ty, vtable);

    cx.vtables().borrow_mut().insert((ty, trait_ref), vtable);
    vtable
}
