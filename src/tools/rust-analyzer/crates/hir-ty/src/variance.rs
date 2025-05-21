//! Module for inferring the variance of type and lifetime parameters. See the [rustc dev guide]
//! chapter for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/variance.html
//!
//! The implementation here differs from rustc. Rustc does a crate wide fixpoint resolution
//! as the algorithm for determining variance is a fixpoint computation with potential cycles that
//! need to be resolved. rust-analyzer does not want a crate-wide analysis though as that would hurt
//! incrementality too much and as such our query is based on a per item basis.
//!
//! This does unfortunately run into the issue that we can run into query cycles which salsa
//! currently does not allow to be resolved via a fixpoint computation. This will likely be resolved
//! by the next salsa version. If not, we will likely have to adapt and go with the rustc approach
//! while installing firewall per item queries to prevent invalidation issues.

use crate::db::HirDatabase;
use crate::generics::{Generics, generics};
use crate::{
    AliasTy, Const, ConstScalar, DynTyExt, GenericArg, GenericArgData, Interner, Lifetime,
    LifetimeData, Ty, TyKind,
};
use chalk_ir::Mutability;
use hir_def::signatures::StructFlags;
use hir_def::{AdtId, GenericDefId, GenericParamId, VariantId};
use std::fmt;
use std::ops::Not;
use stdx::never;
use triomphe::Arc;

pub(crate) fn variances_of(db: &dyn HirDatabase, def: GenericDefId) -> Option<Arc<[Variance]>> {
    tracing::debug!("variances_of(def={:?})", def);
    match def {
        GenericDefId::FunctionId(_) => (),
        GenericDefId::AdtId(adt) => {
            if let AdtId::StructId(id) = adt {
                let flags = &db.struct_signature(id).flags;
                if flags.contains(StructFlags::IS_UNSAFE_CELL) {
                    return Some(Arc::from_iter(vec![Variance::Invariant; 1]));
                } else if flags.contains(StructFlags::IS_PHANTOM_DATA) {
                    return Some(Arc::from_iter(vec![Variance::Covariant; 1]));
                }
            }
        }
        _ => return None,
    }

    let generics = generics(db, def);
    let count = generics.len();
    if count == 0 {
        return None;
    }
    let variances = Context { generics, variances: vec![Variance::Bivariant; count], db }.solve();

    variances.is_empty().not().then(|| Arc::from_iter(variances))
}

// pub(crate) fn variances_of_cycle_fn(
//     _db: &dyn HirDatabase,
//     _result: &Option<Arc<[Variance]>>,
//     _count: u32,
//     _def: GenericDefId,
// ) -> salsa::CycleRecoveryAction<Option<Arc<[Variance]>>> {
//     salsa::CycleRecoveryAction::Iterate
// }

pub(crate) fn variances_of_cycle_initial(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> Option<Arc<[Variance]>> {
    let generics = generics(db, def);
    let count = generics.len();

    if count == 0 {
        return None;
    }
    Some(Arc::from(vec![Variance::Bivariant; count]))
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Variance {
    Covariant,     // T<A> <: T<B> iff A <: B -- e.g., function return type
    Invariant,     // T<A> <: T<B> iff B == A -- e.g., type of mutable cell
    Contravariant, // T<A> <: T<B> iff B <: A -- e.g., function param type
    Bivariant,     // T<A> <: T<B>            -- e.g., unused type parameter
}

impl fmt::Display for Variance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Variance::Covariant => write!(f, "covariant"),
            Variance::Invariant => write!(f, "invariant"),
            Variance::Contravariant => write!(f, "contravariant"),
            Variance::Bivariant => write!(f, "bivariant"),
        }
    }
}

impl Variance {
    /// `a.xform(b)` combines the variance of a context with the
    /// variance of a type with the following meaning. If we are in a
    /// context with variance `a`, and we encounter a type argument in
    /// a position with variance `b`, then `a.xform(b)` is the new
    /// variance with which the argument appears.
    ///
    /// Example 1:
    /// ```ignore (illustrative)
    /// *mut Vec<i32>
    /// ```
    /// Here, the "ambient" variance starts as covariant. `*mut T` is
    /// invariant with respect to `T`, so the variance in which the
    /// `Vec<i32>` appears is `Covariant.xform(Invariant)`, which
    /// yields `Invariant`. Now, the type `Vec<T>` is covariant with
    /// respect to its type argument `T`, and hence the variance of
    /// the `i32` here is `Invariant.xform(Covariant)`, which results
    /// (again) in `Invariant`.
    ///
    /// Example 2:
    /// ```ignore (illustrative)
    /// fn(*const Vec<i32>, *mut Vec<i32)
    /// ```
    /// The ambient variance is covariant. A `fn` type is
    /// contravariant with respect to its parameters, so the variance
    /// within which both pointer types appear is
    /// `Covariant.xform(Contravariant)`, or `Contravariant`. `*const
    /// T` is covariant with respect to `T`, so the variance within
    /// which the first `Vec<i32>` appears is
    /// `Contravariant.xform(Covariant)` or `Contravariant`. The same
    /// is true for its `i32` argument. In the `*mut T` case, the
    /// variance of `Vec<i32>` is `Contravariant.xform(Invariant)`,
    /// and hence the outermost type is `Invariant` with respect to
    /// `Vec<i32>` (and its `i32` argument).
    ///
    /// Source: Figure 1 of "Taming the Wildcards:
    /// Combining Definition- and Use-Site Variance" published in PLDI'11.
    fn xform(self, v: Variance) -> Variance {
        match (self, v) {
            // Figure 1, column 1.
            (Variance::Covariant, Variance::Covariant) => Variance::Covariant,
            (Variance::Covariant, Variance::Contravariant) => Variance::Contravariant,
            (Variance::Covariant, Variance::Invariant) => Variance::Invariant,
            (Variance::Covariant, Variance::Bivariant) => Variance::Bivariant,

            // Figure 1, column 2.
            (Variance::Contravariant, Variance::Covariant) => Variance::Contravariant,
            (Variance::Contravariant, Variance::Contravariant) => Variance::Covariant,
            (Variance::Contravariant, Variance::Invariant) => Variance::Invariant,
            (Variance::Contravariant, Variance::Bivariant) => Variance::Bivariant,

            // Figure 1, column 3.
            (Variance::Invariant, _) => Variance::Invariant,

            // Figure 1, column 4.
            (Variance::Bivariant, _) => Variance::Bivariant,
        }
    }

    fn glb(self, v: Variance) -> Variance {
        // Greatest lower bound of the variance lattice as
        // defined in The Paper:
        //
        //       *
        //    -     +
        //       o
        match (self, v) {
            (Variance::Invariant, _) | (_, Variance::Invariant) => Variance::Invariant,

            (Variance::Covariant, Variance::Contravariant) => Variance::Invariant,
            (Variance::Contravariant, Variance::Covariant) => Variance::Invariant,

            (Variance::Covariant, Variance::Covariant) => Variance::Covariant,

            (Variance::Contravariant, Variance::Contravariant) => Variance::Contravariant,

            (x, Variance::Bivariant) | (Variance::Bivariant, x) => x,
        }
    }

    pub fn invariant(self) -> Self {
        self.xform(Variance::Invariant)
    }

    pub fn covariant(self) -> Self {
        self.xform(Variance::Covariant)
    }

    pub fn contravariant(self) -> Self {
        self.xform(Variance::Contravariant)
    }
}

struct Context<'db> {
    db: &'db dyn HirDatabase,
    generics: Generics,
    variances: Vec<Variance>,
}

impl Context<'_> {
    fn solve(mut self) -> Vec<Variance> {
        tracing::debug!("solve(generics={:?})", self.generics);
        match self.generics.def() {
            GenericDefId::AdtId(adt) => {
                let db = self.db;
                let mut add_constraints_from_variant = |variant| {
                    let subst = self.generics.placeholder_subst(db);
                    for (_, field) in db.field_types(variant).iter() {
                        self.add_constraints_from_ty(
                            &field.clone().substitute(Interner, &subst),
                            Variance::Covariant,
                        );
                    }
                };
                match adt {
                    AdtId::StructId(s) => add_constraints_from_variant(VariantId::StructId(s)),
                    AdtId::UnionId(u) => add_constraints_from_variant(VariantId::UnionId(u)),
                    AdtId::EnumId(e) => {
                        db.enum_variants(e).variants.iter().for_each(|&(variant, _)| {
                            add_constraints_from_variant(VariantId::EnumVariantId(variant))
                        });
                    }
                }
            }
            GenericDefId::FunctionId(f) => {
                let subst = self.generics.placeholder_subst(self.db);
                self.add_constraints_from_sig(
                    self.db
                        .callable_item_signature(f.into())
                        .substitute(Interner, &subst)
                        .params_and_return
                        .iter(),
                    Variance::Covariant,
                );
            }
            _ => {}
        }
        let mut variances = self.variances;

        // Const parameters are always invariant.
        // Make all const parameters invariant.
        for (idx, param) in self.generics.iter_id().enumerate() {
            if let GenericParamId::ConstParamId(_) = param {
                variances[idx] = Variance::Invariant;
            }
        }

        // Functions are permitted to have unused generic parameters: make those invariant.
        if let GenericDefId::FunctionId(_) = self.generics.def() {
            variances
                .iter_mut()
                .filter(|&&mut v| v == Variance::Bivariant)
                .for_each(|v| *v = Variance::Invariant);
        }

        variances
    }

    /// Adds constraints appropriate for an instance of `ty` appearing
    /// in a context with the generics defined in `generics` and
    /// ambient variance `variance`
    fn add_constraints_from_ty(&mut self, ty: &Ty, variance: Variance) {
        tracing::debug!("add_constraints_from_ty(ty={:?}, variance={:?})", ty, variance);
        match ty.kind(Interner) {
            TyKind::Scalar(_) | TyKind::Never | TyKind::Str | TyKind::Foreign(..) => {
                // leaf type -- noop
            }
            TyKind::FnDef(..) | TyKind::Coroutine(..) | TyKind::Closure(..) => {
                never!("Unexpected unnameable type in variance computation: {:?}", ty);
            }
            TyKind::Ref(mutbl, lifetime, ty) => {
                self.add_constraints_from_region(lifetime, variance);
                self.add_constraints_from_mt(ty, *mutbl, variance);
            }
            TyKind::Array(typ, len) => {
                self.add_constraints_from_const(len, variance);
                self.add_constraints_from_ty(typ, variance);
            }
            TyKind::Slice(typ) => {
                self.add_constraints_from_ty(typ, variance);
            }
            TyKind::Raw(mutbl, ty) => {
                self.add_constraints_from_mt(ty, *mutbl, variance);
            }
            TyKind::Tuple(_, subtys) => {
                for subty in subtys.type_parameters(Interner) {
                    self.add_constraints_from_ty(&subty, variance);
                }
            }
            TyKind::Adt(def, args) => {
                self.add_constraints_from_args(def.0.into(), args.as_slice(Interner), variance);
            }
            TyKind::Alias(AliasTy::Opaque(opaque)) => {
                self.add_constraints_from_invariant_args(
                    opaque.substitution.as_slice(Interner),
                    variance,
                );
            }
            TyKind::Alias(AliasTy::Projection(proj)) => {
                self.add_constraints_from_invariant_args(
                    proj.substitution.as_slice(Interner),
                    variance,
                );
            }
            // FIXME: check this
            TyKind::AssociatedType(_, subst) => {
                self.add_constraints_from_invariant_args(subst.as_slice(Interner), variance);
            }
            // FIXME: check this
            TyKind::OpaqueType(_, subst) => {
                self.add_constraints_from_invariant_args(subst.as_slice(Interner), variance);
            }
            TyKind::Dyn(it) => {
                // The type `dyn Trait<T> +'a` is covariant w/r/t `'a`:
                self.add_constraints_from_region(&it.lifetime, variance);

                if let Some(trait_ref) = it.principal() {
                    // Trait are always invariant so we can take advantage of that.
                    self.add_constraints_from_invariant_args(
                        trait_ref
                            .map(|it| it.map(|it| it.substitution.clone()))
                            .substitute(
                                Interner,
                                &[GenericArg::new(
                                    Interner,
                                    chalk_ir::GenericArgData::Ty(TyKind::Error.intern(Interner)),
                                )],
                            )
                            .skip_binders()
                            .as_slice(Interner),
                        variance,
                    );
                }

                // FIXME
                // for projection in data.projection_bounds() {
                //     match projection.skip_binder().term.unpack() {
                //         TyKind::TermKind::Ty(ty) => {
                //             self.add_constraints_from_ty( ty, self.invariant);
                //         }
                //         TyKind::TermKind::Const(c) => {
                //             self.add_constraints_from_const( c, self.invariant)
                //         }
                //     }
                // }
            }

            // Chalk has no params, so use placeholders for now?
            TyKind::Placeholder(index) => {
                let idx = crate::from_placeholder_idx(self.db, *index);
                let index = self.generics.type_or_const_param_idx(idx).unwrap();
                self.constrain(index, variance);
            }
            TyKind::Function(f) => {
                self.add_constraints_from_sig(
                    f.substitution.0.iter(Interner).filter_map(move |p| p.ty(Interner)),
                    variance,
                );
            }
            TyKind::Error => {
                // we encounter this when walking the trait references for object
                // types, where we use Error as the Self type
            }
            TyKind::CoroutineWitness(..) | TyKind::BoundVar(..) | TyKind::InferenceVar(..) => {
                never!("unexpected type encountered in variance inference: {:?}", ty)
            }
        }
    }

    fn add_constraints_from_invariant_args(&mut self, args: &[GenericArg], variance: Variance) {
        let variance_i = variance.invariant();

        for k in args {
            match k.data(Interner) {
                GenericArgData::Lifetime(lt) => self.add_constraints_from_region(lt, variance_i),
                GenericArgData::Ty(ty) => self.add_constraints_from_ty(ty, variance_i),
                GenericArgData::Const(val) => self.add_constraints_from_const(val, variance_i),
            }
        }
    }

    /// Adds constraints appropriate for a nominal type (enum, struct,
    /// object, etc) appearing in a context with ambient variance `variance`
    fn add_constraints_from_args(
        &mut self,
        def_id: GenericDefId,
        args: &[GenericArg],
        variance: Variance,
    ) {
        // We don't record `inferred_starts` entries for empty generics.
        if args.is_empty() {
            return;
        }
        let Some(variances) = self.db.variances_of(def_id) else {
            return;
        };

        for (i, k) in args.iter().enumerate() {
            match k.data(Interner) {
                GenericArgData::Lifetime(lt) => {
                    self.add_constraints_from_region(lt, variance.xform(variances[i]))
                }
                GenericArgData::Ty(ty) => {
                    self.add_constraints_from_ty(ty, variance.xform(variances[i]))
                }
                GenericArgData::Const(val) => self.add_constraints_from_const(val, variance),
            }
        }
    }

    /// Adds constraints appropriate for a const expression `val`
    /// in a context with ambient variance `variance`
    fn add_constraints_from_const(&mut self, c: &Const, variance: Variance) {
        match &c.data(Interner).value {
            chalk_ir::ConstValue::Concrete(c) => {
                if let ConstScalar::UnevaluatedConst(_, subst) = &c.interned {
                    self.add_constraints_from_invariant_args(subst.as_slice(Interner), variance);
                }
            }
            _ => {}
        }
    }

    /// Adds constraints appropriate for a function with signature
    /// `sig` appearing in a context with ambient variance `variance`
    fn add_constraints_from_sig<'a>(
        &mut self,
        mut sig_tys: impl DoubleEndedIterator<Item = &'a Ty>,
        variance: Variance,
    ) {
        let contra = variance.contravariant();
        let Some(output) = sig_tys.next_back() else {
            return never!("function signature has no return type");
        };
        self.add_constraints_from_ty(output, variance);
        for input in sig_tys {
            self.add_constraints_from_ty(input, contra);
        }
    }

    /// Adds constraints appropriate for a region appearing in a
    /// context with ambient variance `variance`
    fn add_constraints_from_region(&mut self, region: &Lifetime, variance: Variance) {
        tracing::debug!(
            "add_constraints_from_region(region={:?}, variance={:?})",
            region,
            variance
        );
        match region.data(Interner) {
            LifetimeData::Placeholder(index) => {
                let idx = crate::lt_from_placeholder_idx(self.db, *index);
                let inferred = self.generics.lifetime_idx(idx).unwrap();
                self.constrain(inferred, variance);
            }
            LifetimeData::Static => {}
            LifetimeData::BoundVar(..) => {
                // Either a higher-ranked region inside of a type or a
                // late-bound function parameter.
                //
                // We do not compute constraints for either of these.
            }
            LifetimeData::Error => {}
            LifetimeData::Phantom(..) | LifetimeData::InferenceVar(..) | LifetimeData::Erased => {
                // We don't expect to see anything but 'static or bound
                // regions when visiting member types or method types.
                never!(
                    "unexpected region encountered in variance \
                      inference: {:?}",
                    region
                );
            }
        }
    }

    /// Adds constraints appropriate for a mutability-type pair
    /// appearing in a context with ambient variance `variance`
    fn add_constraints_from_mt(&mut self, ty: &Ty, mt: Mutability, variance: Variance) {
        self.add_constraints_from_ty(
            ty,
            match mt {
                Mutability::Mut => variance.invariant(),
                Mutability::Not => variance,
            },
        );
    }

    fn constrain(&mut self, index: usize, variance: Variance) {
        tracing::debug!(
            "constrain(index={:?}, variance={:?}, to={:?})",
            index,
            self.variances[index],
            variance
        );
        self.variances[index] = self.variances[index].glb(variance);
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use hir_def::{
        AdtId, GenericDefId, ModuleDefId, hir::generics::GenericParamDataRef, src::HasSource,
    };
    use itertools::Itertools;
    use stdx::format_to;
    use syntax::{AstNode, ast::HasName};
    use test_fixture::WithFixture;

    use hir_def::Lookup;

    use crate::{db::HirDatabase, test_db::TestDB, variance::generics};

    #[test]
    fn phantom_data() {
        check(
            r#"
//- minicore: phantom_data

struct Covariant<A> {
    t: core::marker::PhantomData<A>
}
"#,
            expect![[r#"
                Covariant[A: covariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_types() {
        check(
            r#"
//- minicore: cell

use core::cell::UnsafeCell;

struct InvariantMut<'a,A:'a,B:'a> { //~ ERROR ['a: +, A: o, B: o]
    t: &'a mut (A,B)
}

struct InvariantCell<A> { //~ ERROR [A: o]
    t: UnsafeCell<A>
}

struct InvariantIndirect<A> { //~ ERROR [A: o]
    t: InvariantCell<A>
}

struct Covariant<A> { //~ ERROR [A: +]
    t: A, u: fn() -> A
}

struct Contravariant<A> { //~ ERROR [A: -]
    t: fn(A)
}

enum Enum<A,B,C> { //~ ERROR [A: +, B: -, C: o]
    Foo(Covariant<A>),
    Bar(Contravariant<B>),`
    Zed(Covariant<C>,Contravariant<C>)
}
"#,
            expect![[r#"
                InvariantMut['a: covariant, A: invariant, B: invariant]
                InvariantCell[A: invariant]
                InvariantIndirect[A: invariant]
                Covariant[A: covariant]
                Contravariant[A: contravariant]
                Enum[A: covariant, B: contravariant, C: invariant]
            "#]],
        );
    }

    #[test]
    fn type_resolve_error_two_structs_deep() {
        check(
            r#"
struct Hello<'a> {
    missing: Missing<'a>,
}

struct Other<'a> {
    hello: Hello<'a>,
}
"#,
            expect![[r#"
                Hello['a: bivariant]
                Other['a: bivariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_associated_consts() {
        // FIXME: Should be invariant
        check(
            r#"
trait Trait {
    const Const: usize;
}

struct Foo<T: Trait> { //~ ERROR [T: o]
    field: [u8; <T as Trait>::Const]
}
"#,
            expect![[r#"
                Foo[T: bivariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_associated_types() {
        check(
            r#"
trait Trait<'a> {
    type Type;

    fn method(&'a self) { }
}

struct Foo<'a, T : Trait<'a>> { //~ ERROR ['a: +, T: +]
    field: (T, &'a ())
}

struct Bar<'a, T : Trait<'a>> { //~ ERROR ['a: o, T: o]
    field: <T as Trait<'a>>::Type
}

"#,
            expect![[r#"
                method[Self: contravariant, 'a: contravariant]
                Foo['a: covariant, T: covariant]
                Bar['a: invariant, T: invariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_associated_types2() {
        // FIXME: RPITs have variance, but we can't treat them as their own thing right now
        check(
            r#"
trait Foo {
    type Bar;
}

fn make() -> *const dyn Foo<Bar = &'static u32> {}
"#,
            expect![""],
        );
    }

    #[test]
    fn rustc_test_variance_trait_bounds() {
        check(
            r#"
trait Getter<T> {
    fn get(&self) -> T;
}

trait Setter<T> {
    fn get(&self, _: T);
}

struct TestStruct<U,T:Setter<U>> { //~ ERROR [U: +, T: +]
    t: T, u: U
}

enum TestEnum<U,T:Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    Foo(T)
}

struct TestContraStruct<U,T:Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    t: T
}

struct TestBox<U,T:Getter<U>+Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    t: T
}
"#,
            expect![[r#"
                get[Self: contravariant, T: covariant]
                get[Self: contravariant, T: contravariant]
                TestStruct[U: covariant, T: covariant]
                TestEnum[U: bivariant, T: covariant]
                TestContraStruct[U: bivariant, T: covariant]
                TestBox[U: bivariant, T: covariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_trait_matching() {
        check(
            r#"

trait Get<T> {
    fn get(&self) -> T;
}

struct Cloner<T:Clone> {
    t: T
}

impl<T:Clone> Get<T> for Cloner<T> {
    fn get(&self) -> T {}
}

fn get<'a, G>(get: &G) -> i32
    where G : Get<&'a i32>
{}

fn pick<'b, G>(get: &'b G, if_odd: &'b i32) -> i32
    where G : Get<&'b i32>
{}
"#,
            expect![[r#"
                get[Self: contravariant, T: covariant]
                Cloner[T: covariant]
                get[T: invariant]
                get['a: invariant, G: contravariant]
                pick['b: contravariant, G: contravariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_trait_object_bound() {
        check(
            r#"
enum Option<T> {
    Some(T),
    None
}
trait T { fn foo(&self); }

struct TOption<'a> { //~ ERROR ['a: +]
    v: Option<*const (dyn T + 'a)>,
}
"#,
            expect![[r#"
                Option[T: covariant]
                foo[Self: contravariant]
                TOption['a: covariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_types_bounds() {
        check(
            r#"
//- minicore: send
struct TestImm<A, B> { //~ ERROR [A: +, B: +]
    x: A,
    y: B,
}

struct TestMut<A, B:'static> { //~ ERROR [A: +, B: o]
    x: A,
    y: &'static mut B,
}

struct TestIndirect<A:'static, B:'static> { //~ ERROR [A: +, B: o]
    m: TestMut<A, B>
}

struct TestIndirect2<A:'static, B:'static> { //~ ERROR [A: o, B: o]
    n: TestMut<A, B>,
    m: TestMut<B, A>
}

trait Getter<A> {
    fn get(&self) -> A;
}

trait Setter<A> {
    fn set(&mut self, a: A);
}

struct TestObject<A, R> { //~ ERROR [A: o, R: o]
    n: *const (dyn Setter<A> + Send),
    m: *const (dyn Getter<R> + Send),
}
"#,
            expect![[r#"
                TestImm[A: covariant, B: covariant]
                TestMut[A: covariant, B: invariant]
                TestIndirect[A: covariant, B: invariant]
                TestIndirect2[A: invariant, B: invariant]
                get[Self: contravariant, A: covariant]
                set[Self: invariant, A: contravariant]
                TestObject[A: invariant, R: invariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_unused_region_param() {
        check(
            r#"
struct SomeStruct<'a> { x: u32 } //~ ERROR parameter `'a` is never used
enum SomeEnum<'a> { Nothing } //~ ERROR parameter `'a` is never used
trait SomeTrait<'a> { fn foo(&self); } // OK on traits.
"#,
            expect![[r#"
                SomeStruct['a: bivariant]
                SomeEnum['a: bivariant]
                foo[Self: contravariant, 'a: invariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_unused_type_param() {
        check(
            r#"
//- minicore: sized
struct SomeStruct<A> { x: u32 }
enum SomeEnum<A> { Nothing }
enum ListCell<T> {
    Cons(*const ListCell<T>),
    Nil
}

struct SelfTyAlias<T>(*const Self);
struct WithBounds<T: Sized> {}
struct WithWhereBounds<T> where T: Sized {}
struct WithOutlivesBounds<T: 'static> {}
struct DoubleNothing<T> {
    s: SomeStruct<T>,
}

"#,
            expect![[r#"
                SomeStruct[A: bivariant]
                SomeEnum[A: bivariant]
                ListCell[T: bivariant]
                SelfTyAlias[T: bivariant]
                WithBounds[T: bivariant]
                WithWhereBounds[T: bivariant]
                WithOutlivesBounds[T: bivariant]
                DoubleNothing[T: bivariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_use_contravariant_struct1() {
        check(
            r#"
struct SomeStruct<T>(fn(T));

fn foo<'min,'max>(v: SomeStruct<&'max ()>)
                  -> SomeStruct<&'min ()>
    where 'max : 'min
{}
"#,
            expect![[r#"
                SomeStruct[T: contravariant]
                foo['min: contravariant, 'max: covariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_use_contravariant_struct2() {
        check(
            r#"
struct SomeStruct<T>(fn(T));

fn bar<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{}
"#,
            expect![[r#"
                SomeStruct[T: contravariant]
                bar['min: covariant, 'max: contravariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_use_covariant_struct1() {
        check(
            r#"
struct SomeStruct<T>(T);

fn foo<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{}
"#,
            expect![[r#"
                SomeStruct[T: covariant]
                foo['min: contravariant, 'max: covariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_use_covariant_struct2() {
        check(
            r#"
struct SomeStruct<T>(T);

fn foo<'min,'max>(v: SomeStruct<&'max ()>)
                  -> SomeStruct<&'min ()>
    where 'max : 'min
{}
"#,
            expect![[r#"
                SomeStruct[T: covariant]
                foo['min: covariant, 'max: contravariant]
            "#]],
        );
    }

    #[test]
    fn rustc_test_variance_use_invariant_struct1() {
        check(
            r#"
struct SomeStruct<T>(*mut T);

fn foo<'min,'max>(v: SomeStruct<&'max ()>)
                  -> SomeStruct<&'min ()>
    where 'max : 'min
{}

fn bar<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{}
"#,
            expect![[r#"
                SomeStruct[T: invariant]
                foo['min: invariant, 'max: invariant]
                bar['min: invariant, 'max: invariant]
            "#]],
        );
    }

    #[test]
    fn invalid_arg_counts() {
        check(
            r#"
struct S<T>(T);
struct S2<T>(S<>);
struct S3<T>(S<T, T>);
"#,
            expect![[r#"
                S[T: covariant]
                S2[T: bivariant]
                S3[T: covariant]
            "#]],
        );
    }

    #[test]
    fn prove_fixedpoint() {
        check(
            r#"
struct FixedPoint<T, U, V>(&'static FixedPoint<(), T, U>, V);
"#,
            expect![[r#"
                FixedPoint[T: bivariant, U: bivariant, V: bivariant]
            "#]],
        );
    }

    #[track_caller]
    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expected: Expect) {
        // use tracing_subscriber::{layer::SubscriberExt, Layer};
        // let my_layer = tracing_subscriber::fmt::layer();
        // let _g = tracing::subscriber::set_default(tracing_subscriber::registry().with(
        //     my_layer.with_filter(tracing_subscriber::filter::filter_fn(|metadata| {
        //         metadata.target().starts_with("hir_ty::variance")
        //     })),
        // ));
        let (db, file_id) = TestDB::with_single_file(ra_fixture);

        let mut defs: Vec<GenericDefId> = Vec::new();
        let module = db.module_for_file_opt(file_id.file_id(&db)).unwrap();
        let def_map = module.def_map(&db);
        crate::tests::visit_module(&db, def_map, module.local_id, &mut |it| {
            defs.push(match it {
                ModuleDefId::FunctionId(it) => it.into(),
                ModuleDefId::AdtId(it) => it.into(),
                ModuleDefId::ConstId(it) => it.into(),
                ModuleDefId::TraitId(it) => it.into(),
                ModuleDefId::TraitAliasId(it) => it.into(),
                ModuleDefId::TypeAliasId(it) => it.into(),
                _ => return,
            })
        });
        let defs = defs
            .into_iter()
            .filter_map(|def| {
                Some((
                    def,
                    match def {
                        GenericDefId::FunctionId(it) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::AdtId(AdtId::EnumId(it)) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::AdtId(AdtId::StructId(it)) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::AdtId(AdtId::UnionId(it)) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::TraitId(it) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::TraitAliasId(it) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::TypeAliasId(it) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::ImplId(_) => return None,
                        GenericDefId::ConstId(_) => return None,
                        GenericDefId::StaticId(_) => return None,
                    },
                ))
            })
            .sorted_by_key(|(_, n)| n.syntax().text_range().start());
        let mut res = String::new();
        for (def, name) in defs {
            let Some(variances) = db.variances_of(def) else {
                continue;
            };
            format_to!(
                res,
                "{name}[{}]\n",
                generics(&db, def)
                    .iter()
                    .map(|(_, param)| match param {
                        GenericParamDataRef::TypeParamData(type_param_data) => {
                            type_param_data.name.as_ref().unwrap()
                        }
                        GenericParamDataRef::ConstParamData(const_param_data) =>
                            &const_param_data.name,
                        GenericParamDataRef::LifetimeParamData(lifetime_param_data) => {
                            &lifetime_param_data.name
                        }
                    })
                    .zip_eq(&*variances)
                    .format_with(", ", |(name, var), f| f(&format_args!(
                        "{}: {var}",
                        name.as_str()
                    )))
            );
        }

        expected.assert_eq(&res);
    }
}
