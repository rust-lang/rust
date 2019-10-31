//! Coercion logic. Coercions are certain type conversions that can implicitly
//! happen in certain places, e.g. weakening `&mut` to `&` or deref coercions
//! like going from `&Vec<T>` to `&[T]`.
//!
//! See: https://doc.rust-lang.org/nomicon/coercions.html

use rustc_hash::FxHashMap;

use test_utils::tested_by;

use super::{InferTy, InferenceContext, TypeVarValue};
use crate::{
    db::HirDatabase,
    lang_item::LangItemTarget,
    resolve::Resolver,
    ty::{autoderef, Substs, Ty, TypeCtor, TypeWalk},
    Adt, Mutability,
};

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    /// Unify two types, but may coerce the first one to the second one
    /// using "implicit coercion rules" if needed.
    pub(super) fn coerce(&mut self, from_ty: &Ty, to_ty: &Ty) -> bool {
        let from_ty = self.resolve_ty_shallow(from_ty).into_owned();
        let to_ty = self.resolve_ty_shallow(to_ty);
        self.coerce_inner(from_ty, &to_ty)
    }

    /// Merge two types from different branches, with possible implicit coerce.
    ///
    /// Note that it is only possible that one type are coerced to another.
    /// Coercing both types to another least upper bound type is not possible in rustc,
    /// which will simply result in "incompatible types" error.
    pub(super) fn coerce_merge_branch<'t>(&mut self, ty1: &Ty, ty2: &Ty) -> Ty {
        if self.coerce(ty1, ty2) {
            ty2.clone()
        } else if self.coerce(ty2, ty1) {
            ty1.clone()
        } else {
            tested_by!(coerce_merge_fail_fallback);
            // For incompatible types, we use the latter one as result
            // to be better recovery for `if` without `else`.
            ty2.clone()
        }
    }

    pub(super) fn init_coerce_unsized_map(
        db: &'a D,
        resolver: &Resolver,
    ) -> FxHashMap<(TypeCtor, TypeCtor), usize> {
        let krate = resolver.krate().unwrap();
        let impls = match db.lang_item(krate, "coerce_unsized".into()) {
            Some(LangItemTarget::Trait(trait_)) => db.impls_for_trait(krate, trait_),
            _ => return FxHashMap::default(),
        };

        impls
            .iter()
            .filter_map(|impl_block| {
                // `CoerseUnsized` has one generic parameter for the target type.
                let trait_ref = impl_block.target_trait_ref(db)?;
                let cur_from_ty = trait_ref.substs.0.get(0)?;
                let cur_to_ty = trait_ref.substs.0.get(1)?;

                match (&cur_from_ty, cur_to_ty) {
                    (ty_app!(ctor1, st1), ty_app!(ctor2, st2)) => {
                        // FIXME: We return the first non-equal bound as the type parameter to coerce to unsized type.
                        // This works for smart-pointer-like coercion, which covers all impls from std.
                        st1.iter().zip(st2.iter()).enumerate().find_map(|(i, (ty1, ty2))| {
                            match (ty1, ty2) {
                                (Ty::Param { idx: p1, .. }, Ty::Param { idx: p2, .. })
                                    if p1 != p2 =>
                                {
                                    Some(((*ctor1, *ctor2), i))
                                }
                                _ => None,
                            }
                        })
                    }
                    _ => None,
                }
            })
            .collect()
    }

    fn coerce_inner(&mut self, mut from_ty: Ty, to_ty: &Ty) -> bool {
        match (&from_ty, to_ty) {
            // Never type will make type variable to fallback to Never Type instead of Unknown.
            (ty_app!(TypeCtor::Never), Ty::Infer(InferTy::TypeVar(tv))) => {
                let var = self.new_maybe_never_type_var();
                self.var_unification_table.union_value(*tv, TypeVarValue::Known(var));
                return true;
            }
            (ty_app!(TypeCtor::Never), _) => return true,

            // Trivial cases, this should go after `never` check to
            // avoid infer result type to be never
            _ => {
                if self.unify_inner_trivial(&from_ty, &to_ty) {
                    return true;
                }
            }
        }

        // Pointer weakening and function to pointer
        match (&mut from_ty, to_ty) {
            // `*mut T`, `&mut T, `&T`` -> `*const T`
            // `&mut T` -> `&T`
            // `&mut T` -> `*mut T`
            (ty_app!(c1@TypeCtor::RawPtr(_)), ty_app!(c2@TypeCtor::RawPtr(Mutability::Shared)))
            | (ty_app!(c1@TypeCtor::Ref(_)), ty_app!(c2@TypeCtor::RawPtr(Mutability::Shared)))
            | (ty_app!(c1@TypeCtor::Ref(_)), ty_app!(c2@TypeCtor::Ref(Mutability::Shared)))
            | (ty_app!(c1@TypeCtor::Ref(Mutability::Mut)), ty_app!(c2@TypeCtor::RawPtr(_))) => {
                *c1 = *c2;
            }

            // Illegal mutablity conversion
            (
                ty_app!(TypeCtor::RawPtr(Mutability::Shared)),
                ty_app!(TypeCtor::RawPtr(Mutability::Mut)),
            )
            | (
                ty_app!(TypeCtor::Ref(Mutability::Shared)),
                ty_app!(TypeCtor::Ref(Mutability::Mut)),
            ) => return false,

            // `{function_type}` -> `fn()`
            (ty_app!(TypeCtor::FnDef(_)), ty_app!(TypeCtor::FnPtr { .. })) => {
                match from_ty.callable_sig(self.db) {
                    None => return false,
                    Some(sig) => {
                        let num_args = sig.params_and_return.len() as u16 - 1;
                        from_ty =
                            Ty::apply(TypeCtor::FnPtr { num_args }, Substs(sig.params_and_return));
                    }
                }
            }

            _ => {}
        }

        if let Some(ret) = self.try_coerce_unsized(&from_ty, &to_ty) {
            return ret;
        }

        // Auto Deref if cannot coerce
        match (&from_ty, to_ty) {
            // FIXME: DerefMut
            (ty_app!(TypeCtor::Ref(_), st1), ty_app!(TypeCtor::Ref(_), st2)) => {
                self.unify_autoderef_behind_ref(&st1[0], &st2[0])
            }

            // Otherwise, normal unify
            _ => self.unify(&from_ty, to_ty),
        }
    }

    /// Coerce a type using `from_ty: CoerceUnsized<ty_ty>`
    ///
    /// See: https://doc.rust-lang.org/nightly/std/marker/trait.CoerceUnsized.html
    fn try_coerce_unsized(&mut self, from_ty: &Ty, to_ty: &Ty) -> Option<bool> {
        let (ctor1, st1, ctor2, st2) = match (from_ty, to_ty) {
            (ty_app!(ctor1, st1), ty_app!(ctor2, st2)) => (ctor1, st1, ctor2, st2),
            _ => return None,
        };

        let coerce_generic_index = *self.coerce_unsized_map.get(&(*ctor1, *ctor2))?;

        // Check `Unsize` first
        match self.check_unsize_and_coerce(
            st1.0.get(coerce_generic_index)?,
            st2.0.get(coerce_generic_index)?,
            0,
        ) {
            Some(true) => {}
            ret => return ret,
        }

        let ret = st1
            .iter()
            .zip(st2.iter())
            .enumerate()
            .filter(|&(idx, _)| idx != coerce_generic_index)
            .all(|(_, (ty1, ty2))| self.unify(ty1, ty2));

        Some(ret)
    }

    /// Check if `from_ty: Unsize<to_ty>`, and coerce to `to_ty` if it holds.
    ///
    /// It should not be directly called. It is only used by `try_coerce_unsized`.
    ///
    /// See: https://doc.rust-lang.org/nightly/std/marker/trait.Unsize.html
    fn check_unsize_and_coerce(&mut self, from_ty: &Ty, to_ty: &Ty, depth: usize) -> Option<bool> {
        if depth > 1000 {
            panic!("Infinite recursion in coercion");
        }

        match (&from_ty, &to_ty) {
            // `[T; N]` -> `[T]`
            (ty_app!(TypeCtor::Array, st1), ty_app!(TypeCtor::Slice, st2)) => {
                Some(self.unify(&st1[0], &st2[0]))
            }

            // `T` -> `dyn Trait` when `T: Trait`
            (_, Ty::Dyn(_)) => {
                // FIXME: Check predicates
                Some(true)
            }

            // `(..., T)` -> `(..., U)` when `T: Unsize<U>`
            (
                ty_app!(TypeCtor::Tuple { cardinality: len1 }, st1),
                ty_app!(TypeCtor::Tuple { cardinality: len2 }, st2),
            ) => {
                if len1 != len2 || *len1 == 0 {
                    return None;
                }

                match self.check_unsize_and_coerce(
                    st1.last().unwrap(),
                    st2.last().unwrap(),
                    depth + 1,
                ) {
                    Some(true) => {}
                    ret => return ret,
                }

                let ret = st1[..st1.len() - 1]
                    .iter()
                    .zip(&st2[..st2.len() - 1])
                    .all(|(ty1, ty2)| self.unify(ty1, ty2));

                Some(ret)
            }

            // Foo<..., T, ...> is Unsize<Foo<..., U, ...>> if:
            // - T: Unsize<U>
            // - Foo is a struct
            // - Only the last field of Foo has a type involving T
            // - T is not part of the type of any other fields
            // - Bar<T>: Unsize<Bar<U>>, if the last field of Foo has type Bar<T>
            (
                ty_app!(TypeCtor::Adt(Adt::Struct(struct1)), st1),
                ty_app!(TypeCtor::Adt(Adt::Struct(struct2)), st2),
            ) if struct1 == struct2 => {
                let fields = struct1.fields(self.db);
                let (last_field, prev_fields) = fields.split_last()?;

                // Get the generic parameter involved in the last field.
                let unsize_generic_index = {
                    let mut index = None;
                    let mut multiple_param = false;
                    last_field.ty(self.db).walk(&mut |ty| match ty {
                        &Ty::Param { idx, .. } => {
                            if index.is_none() {
                                index = Some(idx);
                            } else if Some(idx) != index {
                                multiple_param = true;
                            }
                        }
                        _ => {}
                    });

                    if multiple_param {
                        return None;
                    }
                    index?
                };

                // Check other fields do not involve it.
                let mut multiple_used = false;
                prev_fields.iter().for_each(|field| {
                    field.ty(self.db).walk(&mut |ty| match ty {
                        &Ty::Param { idx, .. } if idx == unsize_generic_index => {
                            multiple_used = true
                        }
                        _ => {}
                    })
                });
                if multiple_used {
                    return None;
                }

                let unsize_generic_index = unsize_generic_index as usize;

                // Check `Unsize` first
                match self.check_unsize_and_coerce(
                    st1.get(unsize_generic_index)?,
                    st2.get(unsize_generic_index)?,
                    depth + 1,
                ) {
                    Some(true) => {}
                    ret => return ret,
                }

                // Then unify other parameters
                let ret = st1
                    .iter()
                    .zip(st2.iter())
                    .enumerate()
                    .filter(|&(idx, _)| idx != unsize_generic_index)
                    .all(|(_, (ty1, ty2))| self.unify(ty1, ty2));

                Some(ret)
            }

            _ => None,
        }
    }

    /// Unify `from_ty` to `to_ty` with optional auto Deref
    ///
    /// Note that the parameters are already stripped the outer reference.
    fn unify_autoderef_behind_ref(&mut self, from_ty: &Ty, to_ty: &Ty) -> bool {
        let canonicalized = self.canonicalizer().canonicalize_ty(from_ty.clone());
        let to_ty = self.resolve_ty_shallow(&to_ty);
        // FIXME: Auto DerefMut
        for derefed_ty in
            autoderef::autoderef(self.db, &self.resolver.clone(), canonicalized.value.clone())
        {
            let derefed_ty = canonicalized.decanonicalize_ty(derefed_ty.value);
            match (&*self.resolve_ty_shallow(&derefed_ty), &*to_ty) {
                // Stop when constructor matches.
                (ty_app!(from_ctor, st1), ty_app!(to_ctor, st2)) if from_ctor == to_ctor => {
                    // It will not recurse to `coerce`.
                    return self.unify_substs(st1, st2, 0);
                }
                _ => {}
            }
        }

        false
    }
}
