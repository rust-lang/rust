//! This module provides the built-in trait implementations, e.g. to make
//! closures implement `Fn`.
use hir_def::{expr::Expr, lang_item::LangItemTarget, TraitId, TypeAliasId};
use hir_expand::name::name;
use ra_db::CrateId;

use super::{AssocTyValue, Impl, UnsizeToSuperTraitObjectData};
use crate::{
    db::HirDatabase,
    utils::{all_super_traits, generics},
    ApplicationTy, GenericPredicate, Substs, TraitRef, Ty, TypeCtor,
};

pub(super) struct BuiltinImplData {
    pub num_vars: usize,
    pub trait_ref: TraitRef,
    pub where_clauses: Vec<super::GenericPredicate>,
    pub assoc_ty_values: Vec<AssocTyValue>,
}

pub(super) struct BuiltinImplAssocTyValueData {
    pub impl_: Impl,
    pub assoc_ty_id: TypeAliasId,
    pub num_vars: usize,
    pub value: Ty,
}

pub(super) fn get_builtin_impls(
    db: &impl HirDatabase,
    krate: CrateId,
    ty: &Ty,
    // The first argument for the trait, if present
    arg: &Option<Ty>,
    trait_: TraitId,
    mut callback: impl FnMut(Impl),
) {
    // Note: since impl_datum needs to be infallible, we need to make sure here
    // that we have all prerequisites to build the respective impls.
    if let Ty::Apply(ApplicationTy { ctor: TypeCtor::Closure { def, expr }, .. }) = ty {
        for &fn_trait in [super::FnTrait::FnOnce, super::FnTrait::FnMut, super::FnTrait::Fn].iter()
        {
            if let Some(actual_trait) = get_fn_trait(db, krate, fn_trait) {
                if trait_ == actual_trait {
                    let impl_ = super::ClosureFnTraitImplData { def: *def, expr: *expr, fn_trait };
                    if check_closure_fn_trait_impl_prerequisites(db, krate, impl_) {
                        callback(Impl::ClosureFnTraitImpl(impl_));
                    }
                }
            }
        }
    }

    let unsize_trait = get_unsize_trait(db, krate);
    if let Some(actual_trait) = unsize_trait {
        if trait_ == actual_trait {
            get_builtin_unsize_impls(db, krate, ty, arg, callback);
        }
    }
}

fn get_builtin_unsize_impls(
    db: &impl HirDatabase,
    krate: CrateId,
    ty: &Ty,
    // The first argument for the trait, if present
    arg: &Option<Ty>,
    mut callback: impl FnMut(Impl),
) {
    if !check_unsize_impl_prerequisites(db, krate) {
        return;
    }

    if let Ty::Apply(ApplicationTy { ctor: TypeCtor::Array, .. }) = ty {
        callback(Impl::UnsizeArray);
    }

    if let Some(target_trait) = arg.as_ref().and_then(|t| t.dyn_trait_ref()) {
        if let Some(trait_ref) = ty.dyn_trait_ref() {
            let super_traits = all_super_traits(db, trait_ref.trait_);
            if super_traits.contains(&target_trait.trait_) {
                // callback(Impl::UnsizeToSuperTraitObject(UnsizeToSuperTraitObjectData {
                //     trait_: trait_ref.trait_,
                //     super_trait: target_trait.trait_,
                // }));
            }
        }

        callback(Impl::UnsizeToTraitObject(target_trait.trait_));
    }
}

pub(super) fn impl_datum(db: &impl HirDatabase, krate: CrateId, impl_: Impl) -> BuiltinImplData {
    match impl_ {
        Impl::ImplBlock(_) => unreachable!(),
        Impl::ClosureFnTraitImpl(data) => closure_fn_trait_impl_datum(db, krate, data),
        Impl::UnsizeArray => array_unsize_impl_datum(db, krate),
        Impl::UnsizeToTraitObject(trait_) => trait_object_unsize_impl_datum(db, krate, trait_),
        Impl::UnsizeToSuperTraitObject(data) => {
            super_trait_object_unsize_impl_datum(db, krate, data)
        }
    }
}

pub(super) fn associated_ty_value(
    db: &impl HirDatabase,
    krate: CrateId,
    data: AssocTyValue,
) -> BuiltinImplAssocTyValueData {
    match data {
        AssocTyValue::TypeAlias(_) => unreachable!(),
        AssocTyValue::ClosureFnTraitImplOutput(data) => {
            closure_fn_trait_output_assoc_ty_value(db, krate, data)
        }
    }
}

// Closure Fn trait impls

fn check_closure_fn_trait_impl_prerequisites(
    db: &impl HirDatabase,
    krate: CrateId,
    data: super::ClosureFnTraitImplData,
) -> bool {
    // the respective Fn/FnOnce/FnMut trait needs to exist
    if get_fn_trait(db, krate, data.fn_trait).is_none() {
        return false;
    }

    // FIXME: there are more assumptions that we should probably check here:
    // the traits having no type params, FnOnce being a supertrait

    // the FnOnce trait needs to exist and have an assoc type named Output
    let fn_once_trait = match get_fn_trait(db, krate, super::FnTrait::FnOnce) {
        Some(t) => t,
        None => return false,
    };
    db.trait_data(fn_once_trait).associated_type_by_name(&name![Output]).is_some()
}

fn closure_fn_trait_impl_datum(
    db: &impl HirDatabase,
    krate: CrateId,
    data: super::ClosureFnTraitImplData,
) -> BuiltinImplData {
    // for some closure |X, Y| -> Z:
    // impl<T, U, V> Fn<(T, U)> for closure<fn(T, U) -> V> { Output = V }

    let trait_ = get_fn_trait(db, krate, data.fn_trait) // get corresponding fn trait
        // the existence of the Fn trait has been checked before
        .expect("fn trait for closure impl missing");

    let num_args: u16 = match &db.body(data.def)[data.expr] {
        Expr::Lambda { args, .. } => args.len() as u16,
        _ => {
            log::warn!("closure for closure type {:?} not found", data);
            0
        }
    };

    let arg_ty = Ty::apply(
        TypeCtor::Tuple { cardinality: num_args },
        Substs::builder(num_args as usize).fill_with_bound_vars(0).build(),
    );
    let sig_ty = Ty::apply(
        TypeCtor::FnPtr { num_args },
        Substs::builder(num_args as usize + 1).fill_with_bound_vars(0).build(),
    );

    let self_ty = Ty::apply_one(TypeCtor::Closure { def: data.def, expr: data.expr }, sig_ty);

    let trait_ref = TraitRef {
        trait_,
        substs: Substs::build_for_def(db, trait_).push(self_ty).push(arg_ty).build(),
    };

    let output_ty_id = AssocTyValue::ClosureFnTraitImplOutput(data);

    BuiltinImplData {
        num_vars: num_args as usize + 1,
        trait_ref,
        where_clauses: Vec::new(),
        assoc_ty_values: vec![output_ty_id],
    }
}

fn closure_fn_trait_output_assoc_ty_value(
    db: &impl HirDatabase,
    krate: CrateId,
    data: super::ClosureFnTraitImplData,
) -> BuiltinImplAssocTyValueData {
    let impl_ = Impl::ClosureFnTraitImpl(data);

    let num_args: u16 = match &db.body(data.def)[data.expr] {
        Expr::Lambda { args, .. } => args.len() as u16,
        _ => {
            log::warn!("closure for closure type {:?} not found", data);
            0
        }
    };

    let output_ty = Ty::Bound(num_args.into());

    let fn_once_trait =
        get_fn_trait(db, krate, super::FnTrait::FnOnce).expect("assoc ty value should not exist");

    let output_ty_id = db
        .trait_data(fn_once_trait)
        .associated_type_by_name(&name![Output])
        .expect("assoc ty value should not exist");

    BuiltinImplAssocTyValueData {
        impl_,
        assoc_ty_id: output_ty_id,
        num_vars: num_args as usize + 1,
        value: output_ty,
    }
}

// Array unsizing

fn check_unsize_impl_prerequisites(db: &impl HirDatabase, krate: CrateId) -> bool {
    // the Unsize trait needs to exist and have two type parameters (Self and T)
    let unsize_trait = match get_unsize_trait(db, krate) {
        Some(t) => t,
        None => return false,
    };
    let generic_params = generics(db, unsize_trait.into());
    if generic_params.len() != 2 {
        return false;
    }
    true
}

fn array_unsize_impl_datum(db: &impl HirDatabase, krate: CrateId) -> BuiltinImplData {
    // impl<T> Unsize<[T]> for [T; _]
    // (this can be a single impl because we don't distinguish array sizes currently)

    let trait_ = get_unsize_trait(db, krate) // get unsize trait
        // the existence of the Unsize trait has been checked before
        .expect("Unsize trait missing");

    let var = Ty::Bound(0);
    let substs = Substs::builder(2)
        .push(Ty::apply_one(TypeCtor::Array, var.clone()))
        .push(Ty::apply_one(TypeCtor::Slice, var))
        .build();

    let trait_ref = TraitRef { trait_, substs };

    BuiltinImplData {
        num_vars: 1,
        trait_ref,
        where_clauses: Vec::new(),
        assoc_ty_values: Vec::new(),
    }
}

// Trait object unsizing

fn trait_object_unsize_impl_datum(
    db: &impl HirDatabase,
    krate: CrateId,
    trait_: TraitId,
) -> BuiltinImplData {
    // impl<T, T1, ...> Unsize<dyn Trait<T1, ...>> for T where T: Trait<T1, ...>

    let unsize_trait = get_unsize_trait(db, krate) // get unsize trait
        // the existence of the Unsize trait has been checked before
        .expect("Unsize trait missing");

    let self_ty = Ty::Bound(0);

    let substs = Substs::build_for_def(db, trait_)
        // this fits together nicely: $0 is our self type, and the rest are the type
        // args for the trait
        .fill_with_bound_vars(0)
        .build();
    let trait_ref = TraitRef { trait_, substs };
    // This is both the bound for the `dyn` type, *and* the bound for the impl!
    // This works because the self type for `dyn` is always Ty::Bound(0), which
    // we've also made the parameter for our impl self type.
    let bounds = vec![GenericPredicate::Implemented(trait_ref)];

    let impl_substs = Substs::builder(2).push(self_ty).push(Ty::Dyn(bounds.clone().into())).build();

    let trait_ref = TraitRef { trait_: unsize_trait, substs: impl_substs };

    BuiltinImplData { num_vars: 1, trait_ref, where_clauses: bounds, assoc_ty_values: Vec::new() }
}

fn super_trait_object_unsize_impl_datum(
    db: &impl HirDatabase,
    krate: CrateId,
    _data: UnsizeToSuperTraitObjectData,
) -> BuiltinImplData {
    // impl Unsize<dyn SuperTrait> for dyn Trait

    let unsize_trait = get_unsize_trait(db, krate) // get unsize trait
        // the existence of the Unsize trait has been checked before
        .expect("Unsize trait missing");

    let substs = Substs::builder(2)
        // .push(Ty::Dyn(todo!()))
        // .push(Ty::Dyn(todo!()))
        .build();

    let trait_ref = TraitRef { trait_: unsize_trait, substs };

    BuiltinImplData {
        num_vars: 1,
        trait_ref,
        where_clauses: Vec::new(),
        assoc_ty_values: Vec::new(),
    }
}

fn get_fn_trait(
    db: &impl HirDatabase,
    krate: CrateId,
    fn_trait: super::FnTrait,
) -> Option<TraitId> {
    let target = db.lang_item(krate, fn_trait.lang_item_name().into())?;
    match target {
        LangItemTarget::TraitId(t) => Some(t),
        _ => None,
    }
}

fn get_unsize_trait(db: &impl HirDatabase, krate: CrateId) -> Option<TraitId> {
    let target = db.lang_item(krate, "unsize".into())?;
    match target {
        LangItemTarget::TraitId(t) => Some(t),
        _ => None,
    }
}
