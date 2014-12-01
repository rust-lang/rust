// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generalized type folding mechanism. The setup is a bit convoluted
//! but allows for convenient usage. Let T be an instance of some
//! "foldable type" (one which implements `TypeFoldable`) and F be an
//! instance of a "folder" (a type which implements `TypeFolder`). Then
//! the setup is intended to be:
//!
//!     T.fold_with(F) --calls--> F.fold_T(T) --calls--> super_fold_T(F, T)
//!
//! This way, when you define a new folder F, you can override
//! `fold_T()` to customize the behavior, and invoke `super_fold_T()`
//! to get the original behavior. Meanwhile, to actually fold
//! something, you can just write `T.fold_with(F)`, which is
//! convenient. (Note that `fold_with` will also transparently handle
//! things like a `Vec<T>` where T is foldable and so on.)
//!
//! In this ideal setup, the only function that actually *does*
//! anything is `super_fold_T`, which traverses the type `T`. Moreover,
//! `super_fold_T` should only ever call `T.fold_with()`.
//!
//! In some cases, we follow a degenerate pattern where we do not have
//! a `fold_T` nor `super_fold_T` method. Instead, `T.fold_with`
//! traverses the structure directly. This is suboptimal because the
//! behavior cannot be overriden, but it's much less work to implement.
//! If you ever *do* need an override that doesn't exist, it's not hard
//! to convert the degenerate pattern into the proper thing.

use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::ty::{mod, Ty};
use middle::traits;
use std::rc::Rc;
use syntax::owned_slice::OwnedSlice;
use util::ppaux::Repr;

///////////////////////////////////////////////////////////////////////////
// Two generic traits

/// The TypeFoldable trait is implemented for every type that can be folded.
/// Basically, every type that has a corresponding method in TypeFolder.
pub trait TypeFoldable<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self;
}

/// The TypeFolder trait defines the actual *folding*. There is a
/// method defined for every foldable type. Each of these has a
/// default implementation that does an "identity" fold. Within each
/// identity fold, it should invoke `foo.fold_with(self)` to fold each
/// sub-item.
pub trait TypeFolder<'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx>;

    /// Invoked by the `super_*` routines when we enter a region
    /// binding level (for example, when entering a function
    /// signature). This is used by clients that want to track the
    /// Debruijn index nesting level.
    fn enter_region_binder(&mut self) { }

    /// Invoked by the `super_*` routines when we exit a region
    /// binding level. This is used by clients that want to
    /// track the Debruijn index nesting level.
    fn exit_region_binder(&mut self) { }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        super_fold_ty(self, t)
    }

    fn fold_mt(&mut self, t: &ty::mt<'tcx>) -> ty::mt<'tcx> {
        super_fold_mt(self, t)
    }

    fn fold_trait_ref(&mut self, t: &ty::TraitRef<'tcx>) -> ty::TraitRef<'tcx> {
        super_fold_trait_ref(self, t)
    }

    fn fold_sty(&mut self, sty: &ty::sty<'tcx>) -> ty::sty<'tcx> {
        super_fold_sty(self, sty)
    }

    fn fold_substs(&mut self,
                   substs: &subst::Substs<'tcx>)
                   -> subst::Substs<'tcx> {
        super_fold_substs(self, substs)
    }

    fn fold_fn_sig(&mut self,
                sig: &ty::FnSig<'tcx>)
                -> ty::FnSig<'tcx> {
        super_fold_fn_sig(self, sig)
    }

    fn fold_output(&mut self,
                      output: &ty::FnOutput<'tcx>)
                      -> ty::FnOutput<'tcx> {
        super_fold_output(self, output)
    }

    fn fold_bare_fn_ty(&mut self,
                       fty: &ty::BareFnTy<'tcx>)
                       -> ty::BareFnTy<'tcx>
    {
        super_fold_bare_fn_ty(self, fty)
    }

    fn fold_closure_ty(&mut self,
                       fty: &ty::ClosureTy<'tcx>)
                       -> ty::ClosureTy<'tcx> {
        super_fold_closure_ty(self, fty)
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        r
    }

    fn fold_trait_store(&mut self, s: ty::TraitStore) -> ty::TraitStore {
        super_fold_trait_store(self, s)
    }

    fn fold_existential_bounds(&mut self, s: ty::ExistentialBounds)
                               -> ty::ExistentialBounds {
        super_fold_existential_bounds(self, s)
    }

    fn fold_autoref(&mut self, ar: &ty::AutoRef<'tcx>) -> ty::AutoRef<'tcx> {
        super_fold_autoref(self, ar)
    }

    fn fold_item_substs(&mut self, i: ty::ItemSubsts<'tcx>) -> ty::ItemSubsts<'tcx> {
        super_fold_item_substs(self, i)
    }
}

///////////////////////////////////////////////////////////////////////////
// TypeFoldable implementations.
//
// Ideally, each type should invoke `folder.fold_foo(self)` and
// nothing else. In some cases, though, we haven't gotten around to
// adding methods on the `folder` yet, and thus the folding is
// hard-coded here. This is less-flexible, because folders cannot
// override the behavior, but there are a lot of random types and one
// can easily refactor the folding into the TypeFolder trait as
// needed.

impl<'tcx> TypeFoldable<'tcx> for () {
    fn fold_with<F:TypeFolder<'tcx>>(&self, _: &mut F) -> () {
        ()
    }
}

impl<'tcx, T:TypeFoldable<'tcx>, U:TypeFoldable<'tcx>> TypeFoldable<'tcx> for (T, U) {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> (T, U) {
        (self.0.fold_with(folder), self.1.fold_with(folder))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Option<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Option<T> {
        self.as_ref().map(|t| t.fold_with(folder))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Rc<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Rc<T> {
        Rc::new((**self).fold_with(folder))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Vec<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Vec<T> {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }
}

impl<'tcx, T:TypeFoldable<'tcx>> TypeFoldable<'tcx> for ty::Binder<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Binder<T> {
        folder.enter_region_binder();
        let result = ty::bind(self.value.fold_with(folder));
        folder.exit_region_binder();
        result
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for OwnedSlice<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> OwnedSlice<T> {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for VecPerParamSpace<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> VecPerParamSpace<T> {

        // Things in the Fn space take place under an additional level
        // of region binding relative to the other spaces. This is
        // because those entries are attached to a method, and methods
        // always introduce a level of region binding.

        let result = self.map_enumerated(|(space, index, elem)| {
            if space == subst::FnSpace && index == 0 {
                // enter new level when/if we reach the first thing in fn space
                folder.enter_region_binder();
            }
            elem.fold_with(folder)
        });
        if result.len(subst::FnSpace) > 0 {
            // if there was anything in fn space, exit the region binding level
            folder.exit_region_binder();
        }
        result
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TraitStore {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TraitStore {
        folder.fold_trait_store(*self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for Ty<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Ty<'tcx> {
        folder.fold_ty(*self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::BareFnTy<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::BareFnTy<'tcx> {
        folder.fold_bare_fn_ty(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ClosureTy<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ClosureTy<'tcx> {
        folder.fold_closure_ty(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::mt<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::mt<'tcx> {
        folder.fold_mt(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::FnOutput<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::FnOutput<'tcx> {
        folder.fold_output(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::FnSig<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::FnSig<'tcx> {
        folder.fold_fn_sig(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::sty<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::sty<'tcx> {
        folder.fold_sty(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TraitRef<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TraitRef<'tcx> {
        folder.fold_trait_ref(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Region {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Region {
        folder.fold_region(*self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for subst::Substs<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> subst::Substs<'tcx> {
        folder.fold_substs(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ItemSubsts<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ItemSubsts<'tcx> {
        ty::ItemSubsts {
            substs: self.substs.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::AutoRef<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::AutoRef<'tcx> {
        folder.fold_autoref(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::MethodOrigin<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::MethodOrigin<'tcx> {
        match *self {
            ty::MethodStatic(def_id) => {
                ty::MethodStatic(def_id)
            }
            ty::MethodStaticUnboxedClosure(def_id) => {
                ty::MethodStaticUnboxedClosure(def_id)
            }
            ty::MethodTypeParam(ref param) => {
                ty::MethodTypeParam(ty::MethodParam {
                    trait_ref: param.trait_ref.fold_with(folder),
                    method_num: param.method_num
                })
            }
            ty::MethodTraitObject(ref object) => {
                ty::MethodTraitObject(ty::MethodObject {
                    trait_ref: object.trait_ref.fold_with(folder),
                    object_trait_id: object.object_trait_id,
                    method_num: object.method_num,
                    real_index: object.real_index
                })
            }
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::vtable_origin<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::vtable_origin<'tcx> {
        match *self {
            ty::vtable_static(def_id, ref substs, ref origins) => {
                let r_substs = substs.fold_with(folder);
                let r_origins = origins.fold_with(folder);
                ty::vtable_static(def_id, r_substs, r_origins)
            }
            ty::vtable_param(n, b) => {
                ty::vtable_param(n, b)
            }
            ty::vtable_unboxed_closure(def_id) => {
                ty::vtable_unboxed_closure(def_id)
            }
            ty::vtable_error => {
                ty::vtable_error
            }
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::BuiltinBounds {
    fn fold_with<F: TypeFolder<'tcx>>(&self, _folder: &mut F) -> ty::BuiltinBounds {
        *self
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ExistentialBounds {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ExistentialBounds {
        folder.fold_existential_bounds(*self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ParamBounds<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ParamBounds<'tcx> {
        ty::ParamBounds {
            region_bounds: self.region_bounds.fold_with(folder),
            builtin_bounds: self.builtin_bounds.fold_with(folder),
            trait_bounds: self.trait_bounds.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TypeParameterDef<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TypeParameterDef<'tcx> {
        ty::TypeParameterDef {
            name: self.name,
            def_id: self.def_id,
            space: self.space,
            index: self.index,
            associated_with: self.associated_with,
            bounds: self.bounds.fold_with(folder),
            default: self.default.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::RegionParameterDef {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::RegionParameterDef {
        ty::RegionParameterDef {
            name: self.name,
            def_id: self.def_id,
            space: self.space,
            index: self.index,
            bounds: self.bounds.fold_with(folder)
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Generics<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Generics<'tcx> {
        ty::Generics {
            types: self.types.fold_with(folder),
            regions: self.regions.fold_with(folder),
            predicates: self.predicates.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Predicate<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Predicate<'tcx> {
        match *self {
            ty::Predicate::Trait(ref a) =>
                ty::Predicate::Trait(a.fold_with(folder)),
            ty::Predicate::Equate(ref a, ref b) =>
                ty::Predicate::Equate(a.fold_with(folder),
                                        b.fold_with(folder)),
            ty::Predicate::RegionOutlives(ref a, ref b) =>
                ty::Predicate::RegionOutlives(a.fold_with(folder),
                                                b.fold_with(folder)),
            ty::Predicate::TypeOutlives(ref a, ref b) =>
                ty::Predicate::TypeOutlives(a.fold_with(folder),
                                              b.fold_with(folder)),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::GenericBounds<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::GenericBounds<'tcx> {
        ty::GenericBounds {
            predicates: self.predicates.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::UnsizeKind<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::UnsizeKind<'tcx> {
        match *self {
            ty::UnsizeLength(len) => ty::UnsizeLength(len),
            ty::UnsizeStruct(box ref k, n) => ty::UnsizeStruct(box k.fold_with(folder), n),
            ty::UnsizeVtable(ty::TyTrait{ref principal, bounds}, self_ty) => {
                ty::UnsizeVtable(
                    ty::TyTrait {
                        principal: principal.fold_with(folder),
                        bounds: bounds.fold_with(folder),
                    },
                    self_ty.fold_with(folder))
            }
        }
    }
}

impl<'tcx,O> TypeFoldable<'tcx> for traits::Obligation<'tcx,O>
    where O : TypeFoldable<'tcx>
{
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::Obligation<'tcx, O> {
        traits::Obligation {
            cause: self.cause,
            recursion_depth: self.recursion_depth,
            trait_ref: self.trait_ref.fold_with(folder),
        }
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableImplData<'tcx, N> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::VtableImplData<'tcx, N> {
        traits::VtableImplData {
            impl_def_id: self.impl_def_id,
            substs: self.substs.fold_with(folder),
            nested: self.nested.fold_with(folder),
        }
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableBuiltinData<N> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::VtableBuiltinData<N> {
        traits::VtableBuiltinData {
            nested: self.nested.fold_with(folder),
        }
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::Vtable<'tcx, N> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::Vtable<'tcx, N> {
        match *self {
            traits::VtableImpl(ref v) => traits::VtableImpl(v.fold_with(folder)),
            traits::VtableUnboxedClosure(d, ref s) => {
                traits::VtableUnboxedClosure(d, s.fold_with(folder))
            }
            traits::VtableFnPointer(ref d) => {
                traits::VtableFnPointer(d.fold_with(folder))
            }
            traits::VtableParam(ref p) => traits::VtableParam(p.fold_with(folder)),
            traits::VtableBuiltin(ref d) => traits::VtableBuiltin(d.fold_with(folder)),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::VtableParamData<'tcx> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::VtableParamData<'tcx> {
        traits::VtableParamData {
            bound: self.bound.fold_with(folder),
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// "super" routines: these are the default implementations for TypeFolder.
//
// They should invoke `foo.fold_with()` to do recursive folding.

pub fn super_fold_ty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                t: Ty<'tcx>)
                                                -> Ty<'tcx> {
    let sty = t.sty.fold_with(this);
    ty::mk_t(this.tcx(), sty)
}

pub fn super_fold_substs<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                    substs: &subst::Substs<'tcx>)
                                                    -> subst::Substs<'tcx> {
    let regions = match substs.regions {
        subst::ErasedRegions => {
            subst::ErasedRegions
        }
        subst::NonerasedRegions(ref regions) => {
            subst::NonerasedRegions(regions.fold_with(this))
        }
    };

    subst::Substs { regions: regions,
                    types: substs.types.fold_with(this) }
}

pub fn super_fold_fn_sig<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                    sig: &ty::FnSig<'tcx>)
                                                    -> ty::FnSig<'tcx>
{
    this.enter_region_binder();
    let result = super_fold_fn_sig_contents(this, sig);
    this.exit_region_binder();
    result
}

pub fn super_fold_fn_sig_contents<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                             sig: &ty::FnSig<'tcx>)
                                                             -> ty::FnSig<'tcx>
{
    ty::FnSig { inputs: sig.inputs.fold_with(this),
                output: sig.output.fold_with(this),
                variadic: sig.variadic }
}

pub fn super_fold_output<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                    output: &ty::FnOutput<'tcx>)
                                                    -> ty::FnOutput<'tcx> {
    match *output {
        ty::FnConverging(ref ty) => ty::FnConverging(ty.fold_with(this)),
        ty::FnDiverging => ty::FnDiverging
    }
}

pub fn super_fold_bare_fn_ty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                        fty: &ty::BareFnTy<'tcx>)
                                                        -> ty::BareFnTy<'tcx>
{
    ty::BareFnTy { sig: fty.sig.fold_with(this),
                   abi: fty.abi,
                   unsafety: fty.unsafety }
}

pub fn super_fold_closure_ty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                        fty: &ty::ClosureTy<'tcx>)
                                                        -> ty::ClosureTy<'tcx>
{
    ty::ClosureTy {
        store: fty.store.fold_with(this),
        sig: fty.sig.fold_with(this),
        unsafety: fty.unsafety,
        onceness: fty.onceness,
        bounds: fty.bounds.fold_with(this),
        abi: fty.abi,
    }
}

pub fn super_fold_trait_ref<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                       t: &ty::TraitRef<'tcx>)
                                                       -> ty::TraitRef<'tcx>
{
    this.enter_region_binder();
    let result = super_fold_trait_ref_contents(this, t);
    this.exit_region_binder();
    result
}

pub fn super_fold_trait_ref_contents<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                                t: &ty::TraitRef<'tcx>)
                                                                -> ty::TraitRef<'tcx>
{
    ty::TraitRef {
        def_id: t.def_id,
        substs: t.substs.fold_with(this),
    }
}

pub fn super_fold_mt<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                mt: &ty::mt<'tcx>)
                                                -> ty::mt<'tcx> {
    ty::mt {ty: mt.ty.fold_with(this),
            mutbl: mt.mutbl}
}

pub fn super_fold_sty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                 sty: &ty::sty<'tcx>)
                                                 -> ty::sty<'tcx> {
    match *sty {
        ty::ty_uniq(typ) => {
            ty::ty_uniq(typ.fold_with(this))
        }
        ty::ty_ptr(ref tm) => {
            ty::ty_ptr(tm.fold_with(this))
        }
        ty::ty_vec(typ, sz) => {
            ty::ty_vec(typ.fold_with(this), sz)
        }
        ty::ty_open(typ) => {
            ty::ty_open(typ.fold_with(this))
        }
        ty::ty_enum(tid, ref substs) => {
            ty::ty_enum(tid, substs.fold_with(this))
        }
        ty::ty_trait(box ty::TyTrait { ref principal, bounds }) => {
            ty::ty_trait(box ty::TyTrait {
                principal: (*principal).fold_with(this),
                bounds: bounds.fold_with(this),
            })
        }
        ty::ty_tup(ref ts) => {
            ty::ty_tup(ts.fold_with(this))
        }
        ty::ty_bare_fn(ref f) => {
            ty::ty_bare_fn(f.fold_with(this))
        }
        ty::ty_closure(ref f) => {
            ty::ty_closure(box f.fold_with(this))
        }
        ty::ty_rptr(r, ref tm) => {
            ty::ty_rptr(r.fold_with(this), tm.fold_with(this))
        }
        ty::ty_struct(did, ref substs) => {
            ty::ty_struct(did, substs.fold_with(this))
        }
        ty::ty_unboxed_closure(did, ref region, ref substs) => {
            ty::ty_unboxed_closure(did, region.fold_with(this), substs.fold_with(this))
        }
        ty::ty_bool | ty::ty_char | ty::ty_str |
        ty::ty_int(_) | ty::ty_uint(_) | ty::ty_float(_) |
        ty::ty_err | ty::ty_infer(_) |
        ty::ty_param(..) => {
            (*sty).clone()
        }
    }
}

pub fn super_fold_trait_store<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                         trait_store: ty::TraitStore)
                                                         -> ty::TraitStore {
    match trait_store {
        ty::UniqTraitStore => ty::UniqTraitStore,
        ty::RegionTraitStore(r, m) => {
            ty::RegionTraitStore(r.fold_with(this), m)
        }
    }
}

pub fn super_fold_existential_bounds<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                                bounds: ty::ExistentialBounds)
                                                                -> ty::ExistentialBounds {
    ty::ExistentialBounds {
        region_bound: bounds.region_bound.fold_with(this),
        builtin_bounds: bounds.builtin_bounds,
    }
}

pub fn super_fold_autoref<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                     autoref: &ty::AutoRef<'tcx>)
                                                     -> ty::AutoRef<'tcx>
{
    match *autoref {
        ty::AutoPtr(r, m, None) => ty::AutoPtr(this.fold_region(r), m, None),
        ty::AutoPtr(r, m, Some(ref a)) => {
            ty::AutoPtr(this.fold_region(r), m, Some(box super_fold_autoref(this, &**a)))
        }
        ty::AutoUnsafe(m, None) => ty::AutoUnsafe(m, None),
        ty::AutoUnsafe(m, Some(ref a)) => {
            ty::AutoUnsafe(m, Some(box super_fold_autoref(this, &**a)))
        }
        ty::AutoUnsize(ref k) => ty::AutoUnsize(k.fold_with(this)),
        ty::AutoUnsizeUniq(ref k) => ty::AutoUnsizeUniq(k.fold_with(this)),
    }
}

pub fn super_fold_item_substs<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                         substs: ty::ItemSubsts<'tcx>)
                                                         -> ty::ItemSubsts<'tcx>
{
    ty::ItemSubsts {
        substs: substs.substs.fold_with(this),
    }
}

///////////////////////////////////////////////////////////////////////////
// Higher-ranked things

/// Designates a "binder" for late-bound regions.
pub trait HigherRankedFoldable<'tcx>: Repr<'tcx> {
    /// Folds the contents of `self`, ignoring the region binder created
    /// by `self`.
    fn fold_contents<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self;
}

impl<'tcx> HigherRankedFoldable<'tcx> for ty::FnSig<'tcx> {
    fn fold_contents<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::FnSig<'tcx> {
        super_fold_fn_sig_contents(folder, self)
    }
}

impl<'tcx> HigherRankedFoldable<'tcx> for ty::TraitRef<'tcx> {
    fn fold_contents<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TraitRef<'tcx> {
        super_fold_trait_ref_contents(folder, self)
    }
}

impl<'tcx, T:TypeFoldable<'tcx>+Repr<'tcx>> HigherRankedFoldable<'tcx> for ty::Binder<T> {
    fn fold_contents<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Binder<T> {
        ty::bind(self.value.fold_with(folder))
    }
}

impl<'tcx, T:HigherRankedFoldable<'tcx>> HigherRankedFoldable<'tcx> for Rc<T> {
    fn fold_contents<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Rc<T> {
        Rc::new((**self).fold_contents(folder))
    }
}

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'a, 'tcx: 'a, F> where F: FnMut(Ty<'tcx>) -> Ty<'tcx> {
    pub tcx: &'a ty::ctxt<'tcx>,
    pub fldop: F,
}

impl<'a, 'tcx, F> TypeFolder<'tcx> for BottomUpFolder<'a, 'tcx, F> where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
{
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> { self.tcx }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let t1 = super_fold_ty(self, ty);
        (self.fldop)(t1)
    }
}

///////////////////////////////////////////////////////////////////////////
// Region folder

/// Folds over the substructure of a type, visiting its component
/// types and all regions that occur *free* within it.
///
/// That is, `Ty` can contain function or method types that bind
/// regions at the call site (`ReLateBound`), and occurrences of
/// regions (aka "lifetimes") that are bound within a type are not
/// visited by this folder; only regions that occur free will be
/// visited by `fld_r`.
///
/// (The distinction between "free" and "bound" is represented by
/// keeping track of each `FnSig` in the lexical context of the
/// current position of the fold.)
pub struct RegionFolder<'a, 'tcx: 'a, F> where F: FnMut(ty::Region, uint) -> ty::Region {
    tcx: &'a ty::ctxt<'tcx>,
    current_depth: uint,
    fld_r: F,
}

impl<'a, 'tcx, F> RegionFolder<'a, 'tcx, F> where F: FnMut(ty::Region, uint) -> ty::Region {
    pub fn new(tcx: &'a ty::ctxt<'tcx>, fld_r: F) -> RegionFolder<'a, 'tcx, F> {
        RegionFolder {
            tcx: tcx,
            current_depth: 1,
            fld_r: fld_r,
        }
    }
}

pub fn collect_regions<'tcx,T>(tcx: &ty::ctxt<'tcx>, value: &T) -> Vec<ty::Region>
    where T : TypeFoldable<'tcx>
{
    let mut vec = Vec::new();
    {
        let mut folder = RegionFolder::new(tcx, |r, _| { vec.push(r); r });
        value.fold_with(&mut folder);
    }
    vec
}

impl<'a, 'tcx, F> TypeFolder<'tcx> for RegionFolder<'a, 'tcx, F> where
    F: FnMut(ty::Region, uint) -> ty::Region,
{
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> { self.tcx }

    fn enter_region_binder(&mut self) {
        self.current_depth += 1;
    }

    fn exit_region_binder(&mut self) {
        self.current_depth -= 1;
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match r {
            ty::ReLateBound(debruijn, _) if debruijn.depth < self.current_depth => {
                debug!("RegionFolder.fold_region({}) skipped bound region (current depth={})",
                       r.repr(self.tcx()), self.current_depth);
                r
            }
            _ => {
                debug!("RegionFolder.fold_region({}) folding free region (current_depth={})",
                       r.repr(self.tcx()), self.current_depth);
                (self.fld_r)(r, self.current_depth)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Region eraser
//
// Replaces all free regions with 'static. Useful in trans.

pub struct RegionEraser<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
}

pub fn erase_regions<'tcx, T: TypeFoldable<'tcx>>(tcx: &ty::ctxt<'tcx>, t: T) -> T {
    let mut eraser = RegionEraser { tcx: tcx };
    t.fold_with(&mut eraser)
}

impl<'a, 'tcx> TypeFolder<'tcx> for RegionEraser<'a, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> { self.tcx }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match r {
            ty::ReLateBound(..) | ty::ReEarlyBound(..) => r,
            _ => ty::ReStatic
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Region shifter
//
// Shifts the De Bruijn indices on all escaping bound regions by a
// fixed amount. Useful in substitution or when otherwise introducing
// a binding level that is not intended to capture the existing bound
// regions. See comment on `shift_regions_through_binders` method in
// `subst.rs` for more details.

pub fn shift_region(region: ty::Region, amount: uint) -> ty::Region {
    match region {
        ty::ReLateBound(debruijn, br) => {
            ty::ReLateBound(debruijn.shifted(amount), br)
        }
        _ => {
            region
        }
    }
}

pub fn shift_regions<'tcx, T:TypeFoldable<'tcx>+Repr<'tcx>>(tcx: &ty::ctxt<'tcx>,
                                                            amount: uint, value: &T) -> T {
    debug!("shift_regions(value={}, amount={})",
           value.repr(tcx), amount);

    value.fold_with(&mut RegionFolder::new(tcx, |region, _current_depth| {
        shift_region(region, amount)
    }))
}

