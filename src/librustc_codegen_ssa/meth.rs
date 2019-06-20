use rustc_target::abi::call::FnType;

use crate::callee;
use crate::traits::*;

use rustc::ty::{self, Ty, Instance};

#[derive(Copy, Clone, Debug)]
pub struct VirtualIndex(u64);

pub const DESTRUCTOR: VirtualIndex = VirtualIndex(0);
pub const SIZE: VirtualIndex = VirtualIndex(1);
pub const ALIGN: VirtualIndex = VirtualIndex(2);

impl<'a, 'tcx> VirtualIndex {
    pub fn from_index(index: usize) -> Self {
        VirtualIndex(index as u64 + 3)
    }

    pub fn get_fn<Bx: BuilderMethods<'a, 'tcx>>(
        self,
        bx: &mut Bx,
        llvtable: Bx::Value,
        fn_ty: &FnType<'tcx, Ty<'tcx>>
    ) -> Bx::Value {
        // Load the data pointer from the object.
        debug!("get_fn({:?}, {:?})", llvtable, self);

        let llvtable = bx.pointercast(
            llvtable,
            bx.type_ptr_to(bx.fn_ptr_backend_type(fn_ty))
        );
        let ptr_align = bx.tcx().data_layout.pointer_align.abi;
        let gep = bx.inbounds_gep(llvtable, &[bx.const_usize(self.0)]);
        let ptr = bx.load(gep, ptr_align);
        bx.nonnull_metadata(ptr);
        // Vtable loads are invariant
        bx.set_invariant_load(ptr);
        ptr
    }

    pub fn get_usize<Bx: BuilderMethods<'a, 'tcx>>(
        self,
        bx: &mut Bx,
        llvtable: Bx::Value
    ) -> Bx::Value {
        // Load the data pointer from the object.
        debug!("get_int({:?}, {:?})", llvtable, self);

        let llvtable = bx.pointercast(llvtable, bx.type_ptr_to(bx.type_isize()));
        let usize_align = bx.tcx().data_layout.pointer_align.abi;
        let gep = bx.inbounds_gep(llvtable, &[bx.const_usize(self.0)]);
        let ptr = bx.load(gep, usize_align);
        // Vtable loads are invariant
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
/// `trait_ref` would map `T:Trait`.
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

    // Not in the cache. Build it.
    let nullptr = cx.const_null(cx.type_i8p());

    let methods_root;
    let methods = if let Some(trait_ref) = trait_ref {
        methods_root = tcx.vtable_methods(trait_ref.with_self_ty(tcx, ty));
        methods_root.iter()
    } else {
        (&[]).iter()
    };

    let methods = methods.cloned().map(|opt_mth| {
        opt_mth.map_or(nullptr, |(def_id, substs)| {
            callee::resolve_and_get_fn_for_vtable(cx, def_id, substs)
        })
    });

    let layout = cx.layout_of(ty);
    // /////////////////////////////////////////////////////////////////////////////////////////////
    // If you touch this code, be sure to also make the corresponding changes to
    // `get_vtable` in rust_mir/interpret/traits.rs
    // /////////////////////////////////////////////////////////////////////////////////////////////
    let components: Vec<_> = [
        cx.get_fn(Instance::resolve_drop_in_place(cx.tcx(), ty)),
        cx.const_usize(layout.size.bytes()),
        cx.const_usize(layout.align.abi.bytes())
    ].iter().cloned().chain(methods).collect();

    let vtable_const = cx.const_struct(&components, false);
    let align = cx.data_layout().pointer_align.abi;
    let vtable = cx.static_addr_of(vtable_const, align, Some("vtable"));

    cx.create_vtable_metadata(ty, vtable);

    cx.vtables().borrow_mut().insert((ty, trait_ref), vtable);
    vtable
}
