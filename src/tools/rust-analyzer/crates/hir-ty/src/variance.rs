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

use hir_def::{AdtId, GenericDefId, GenericParamId, VariantId, signatures::StructFlags};
use rustc_ast_ir::Mutability;
use rustc_type_ir::{
    Variance,
    inherent::{AdtDef, IntoKind, SliceLike},
};
use stdx::never;

use crate::{
    db::HirDatabase,
    generics::{Generics, generics},
    next_solver::{
        Const, ConstKind, DbInterner, ExistentialPredicate, GenericArg, GenericArgs, Region,
        RegionKind, Term, Ty, TyKind, VariancesOf,
    },
};

pub(crate) fn variances_of(db: &dyn HirDatabase, def: GenericDefId) -> VariancesOf<'_> {
    tracing::debug!("variances_of(def={:?})", def);
    let interner = DbInterner::new_no_crate(db);
    match def {
        GenericDefId::FunctionId(_) => (),
        GenericDefId::AdtId(adt) => {
            if let AdtId::StructId(id) = adt {
                let flags = &db.struct_signature(id).flags;
                if flags.contains(StructFlags::IS_UNSAFE_CELL) {
                    return VariancesOf::new_from_iter(interner, [Variance::Invariant]);
                } else if flags.contains(StructFlags::IS_PHANTOM_DATA) {
                    return VariancesOf::new_from_iter(interner, [Variance::Covariant]);
                }
            }
        }
        _ => return VariancesOf::new_from_iter(interner, []),
    }

    let generics = generics(db, def);
    let count = generics.len();
    if count == 0 {
        return VariancesOf::new_from_iter(interner, []);
    }
    let mut variances =
        Context { generics, variances: vec![Variance::Bivariant; count], db }.solve();

    // FIXME(next-solver): This is *not* the correct behavior. I don't know if it has an actual effect,
    // since bivariance is prohibited in Rust, but rustc definitely does not fallback bivariance.
    // So why do we do this? Because, with the new solver, the effects of bivariance are catastrophic:
    // it leads to not relating types properly, and to very, very hard to debug bugs (speaking from experience).
    // Furthermore, our variance infra is known to not handle cycles properly. Therefore, at least until we fix
    // cycles, and perhaps forever at least for out tests, not allowing bivariance makes sense.
    // Why specifically invariance? I don't have a strong reason, mainly that invariance is a stronger relationship
    // (therefore, less room for mistakes) and that IMO incorrect covariance can be more problematic that incorrect
    // bivariance, at least while we don't handle lifetimes anyway.
    for variance in &mut variances {
        if *variance == Variance::Bivariant {
            *variance = Variance::Invariant;
        }
    }

    VariancesOf::new_from_iter(interner, variances)
}

// pub(crate) fn variances_of_cycle_fn(
//     _db: &dyn HirDatabase,
//     _result: &Option<Arc<[Variance]>>,
//     _count: u32,
//     _def: GenericDefId,
// ) -> salsa::CycleRecoveryAction<Option<Arc<[Variance]>>> {
//     salsa::CycleRecoveryAction::Iterate
// }

fn glb(v1: Variance, v2: Variance) -> Variance {
    // Greatest lower bound of the variance lattice as defined in The Paper:
    //
    //       *
    //    -     +
    //       o
    match (v1, v2) {
        (Variance::Invariant, _) | (_, Variance::Invariant) => Variance::Invariant,

        (Variance::Covariant, Variance::Contravariant) => Variance::Invariant,
        (Variance::Contravariant, Variance::Covariant) => Variance::Invariant,

        (Variance::Covariant, Variance::Covariant) => Variance::Covariant,

        (Variance::Contravariant, Variance::Contravariant) => Variance::Contravariant,

        (x, Variance::Bivariant) | (Variance::Bivariant, x) => x,
    }
}

pub(crate) fn variances_of_cycle_initial(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> VariancesOf<'_> {
    let interner = DbInterner::new_no_crate(db);
    let generics = generics(db, def);
    let count = generics.len();

    // FIXME(next-solver): Returns `Invariance` and not `Bivariance` here, see the comment in the main query.
    VariancesOf::new_from_iter(interner, std::iter::repeat_n(Variance::Invariant, count))
}

struct Context<'db> {
    db: &'db dyn HirDatabase,
    generics: Generics,
    variances: Vec<Variance>,
}

impl<'db> Context<'db> {
    fn solve(mut self) -> Vec<Variance> {
        tracing::debug!("solve(generics={:?})", self.generics);
        match self.generics.def() {
            GenericDefId::AdtId(adt) => {
                let db = self.db;
                let mut add_constraints_from_variant = |variant| {
                    for (_, field) in db.field_types(variant).iter() {
                        self.add_constraints_from_ty(
                            field.instantiate_identity(),
                            Variance::Covariant,
                        );
                    }
                };
                match adt {
                    AdtId::StructId(s) => add_constraints_from_variant(VariantId::StructId(s)),
                    AdtId::UnionId(u) => add_constraints_from_variant(VariantId::UnionId(u)),
                    AdtId::EnumId(e) => {
                        e.enum_variants(db).variants.iter().for_each(|&(variant, _, _)| {
                            add_constraints_from_variant(VariantId::EnumVariantId(variant))
                        });
                    }
                }
            }
            GenericDefId::FunctionId(f) => {
                let sig =
                    self.db.callable_item_signature(f.into()).instantiate_identity().skip_binder();
                self.add_constraints_from_sig(sig.inputs_and_output.iter(), Variance::Covariant);
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
    fn add_constraints_from_ty(&mut self, ty: Ty<'db>, variance: Variance) {
        tracing::debug!("add_constraints_from_ty(ty={:?}, variance={:?})", ty, variance);
        match ty.kind() {
            TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_)
            | TyKind::Char
            | TyKind::Bool
            | TyKind::Never
            | TyKind::Str
            | TyKind::Foreign(..) => {
                // leaf type -- noop
            }
            TyKind::FnDef(..)
            | TyKind::Coroutine(..)
            | TyKind::CoroutineClosure(..)
            | TyKind::Closure(..) => {
                never!("Unexpected unnameable type in variance computation: {:?}", ty);
            }
            TyKind::Ref(lifetime, ty, mutbl) => {
                self.add_constraints_from_region(lifetime, variance);
                self.add_constraints_from_mt(ty, mutbl, variance);
            }
            TyKind::Array(typ, len) => {
                self.add_constraints_from_const(len);
                self.add_constraints_from_ty(typ, variance);
            }
            TyKind::Slice(typ) => {
                self.add_constraints_from_ty(typ, variance);
            }
            TyKind::RawPtr(ty, mutbl) => {
                self.add_constraints_from_mt(ty, mutbl, variance);
            }
            TyKind::Tuple(subtys) => {
                for subty in subtys {
                    self.add_constraints_from_ty(subty, variance);
                }
            }
            TyKind::Adt(def, args) => {
                self.add_constraints_from_args(def.def_id().0.into(), args, variance);
            }
            TyKind::Alias(_, alias) => {
                // FIXME: Probably not correct wrt. opaques.
                self.add_constraints_from_invariant_args(alias.args);
            }
            TyKind::Dynamic(bounds, region) => {
                // The type `dyn Trait<T> +'a` is covariant w/r/t `'a`:
                self.add_constraints_from_region(region, variance);

                for bound in bounds {
                    match bound.skip_binder() {
                        ExistentialPredicate::Trait(trait_ref) => {
                            self.add_constraints_from_invariant_args(trait_ref.args)
                        }
                        ExistentialPredicate::Projection(projection) => {
                            self.add_constraints_from_invariant_args(projection.args);
                            match projection.term {
                                Term::Ty(ty) => {
                                    self.add_constraints_from_ty(ty, Variance::Invariant)
                                }
                                Term::Const(konst) => self.add_constraints_from_const(konst),
                            }
                        }
                        ExistentialPredicate::AutoTrait(_) => {}
                    }
                }
            }

            // Chalk has no params, so use placeholders for now?
            TyKind::Param(param) => self.constrain(param.index as usize, variance),
            TyKind::FnPtr(sig, _) => {
                self.add_constraints_from_sig(sig.skip_binder().inputs_and_output.iter(), variance);
            }
            TyKind::Error(_) => {
                // we encounter this when walking the trait references for object
                // types, where we use Error as the Self type
            }
            TyKind::Bound(..) => {}
            TyKind::CoroutineWitness(..)
            | TyKind::Placeholder(..)
            | TyKind::Infer(..)
            | TyKind::UnsafeBinder(..)
            | TyKind::Pat(..) => {
                never!("unexpected type encountered in variance inference: {:?}", ty)
            }
        }
    }

    fn add_constraints_from_invariant_args(&mut self, args: GenericArgs<'db>) {
        for k in args.iter() {
            match k {
                GenericArg::Lifetime(lt) => {
                    self.add_constraints_from_region(lt, Variance::Invariant)
                }
                GenericArg::Ty(ty) => self.add_constraints_from_ty(ty, Variance::Invariant),
                GenericArg::Const(val) => self.add_constraints_from_const(val),
            }
        }
    }

    /// Adds constraints appropriate for a nominal type (enum, struct,
    /// object, etc) appearing in a context with ambient variance `variance`
    fn add_constraints_from_args(
        &mut self,
        def_id: GenericDefId,
        args: GenericArgs<'db>,
        variance: Variance,
    ) {
        if args.is_empty() {
            return;
        }
        let variances = self.db.variances_of(def_id);

        for (k, v) in args.iter().zip(variances) {
            match k {
                GenericArg::Lifetime(lt) => self.add_constraints_from_region(lt, variance.xform(v)),
                GenericArg::Ty(ty) => self.add_constraints_from_ty(ty, variance.xform(v)),
                GenericArg::Const(val) => self.add_constraints_from_const(val),
            }
        }
    }

    /// Adds constraints appropriate for a const expression `val`
    /// in a context with ambient variance `variance`
    fn add_constraints_from_const(&mut self, c: Const<'db>) {
        match c.kind() {
            ConstKind::Unevaluated(c) => self.add_constraints_from_invariant_args(c.args),
            _ => {}
        }
    }

    /// Adds constraints appropriate for a function with signature
    /// `sig` appearing in a context with ambient variance `variance`
    fn add_constraints_from_sig(
        &mut self,
        mut sig_tys: impl DoubleEndedIterator<Item = Ty<'db>>,
        variance: Variance,
    ) {
        let contra = variance.xform(Variance::Contravariant);
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
    fn add_constraints_from_region(&mut self, region: Region<'db>, variance: Variance) {
        tracing::debug!(
            "add_constraints_from_region(region={:?}, variance={:?})",
            region,
            variance
        );
        match region.kind() {
            RegionKind::ReEarlyParam(param) => self.constrain(param.index as usize, variance),
            RegionKind::ReStatic => {}
            RegionKind::ReBound(..) => {
                // Either a higher-ranked region inside of a type or a
                // late-bound function parameter.
                //
                // We do not compute constraints for either of these.
            }
            RegionKind::ReError(_) => {}
            RegionKind::ReLateParam(..)
            | RegionKind::RePlaceholder(..)
            | RegionKind::ReVar(..)
            | RegionKind::ReErased => {
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
    fn add_constraints_from_mt(&mut self, ty: Ty<'db>, mt: Mutability, variance: Variance) {
        self.add_constraints_from_ty(
            ty,
            match mt {
                Mutability::Mut => Variance::Invariant,
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
        self.variances[index] = glb(self.variances[index], variance);
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use hir_def::{
        AdtId, GenericDefId, ModuleDefId, hir::generics::GenericParamDataRef, src::HasSource,
    };
    use itertools::Itertools;
    use rustc_type_ir::{Variance, inherent::SliceLike};
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
                Hello['a: invariant]
                Other['a: invariant]
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
                Foo[T: invariant]
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
                TestEnum[U: invariant, T: covariant]
                TestContraStruct[U: invariant, T: covariant]
                TestBox[U: invariant, T: covariant]
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
                SomeStruct['a: invariant]
                SomeEnum['a: invariant]
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
                SomeStruct[A: invariant]
                SomeEnum[A: invariant]
                ListCell[T: invariant]
                SelfTyAlias[T: invariant]
                WithBounds[T: invariant]
                WithWhereBounds[T: invariant]
                WithOutlivesBounds[T: invariant]
                DoubleNothing[T: invariant]
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
                S2[T: invariant]
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
                FixedPoint[T: invariant, U: invariant, V: invariant]
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

        crate::attach_db(&db, || {
            let mut defs: Vec<GenericDefId> = Vec::new();
            let module = db.module_for_file_opt(file_id.file_id(&db)).unwrap();
            let def_map = module.def_map(&db);
            crate::tests::visit_module(&db, def_map, module, &mut |it| {
                defs.push(match it {
                    ModuleDefId::FunctionId(it) => it.into(),
                    ModuleDefId::AdtId(it) => it.into(),
                    ModuleDefId::ConstId(it) => it.into(),
                    ModuleDefId::TraitId(it) => it.into(),
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
                            GenericDefId::TraitId(_)
                            | GenericDefId::TypeAliasId(_)
                            | GenericDefId::ImplId(_)
                            | GenericDefId::ConstId(_)
                            | GenericDefId::StaticId(_) => return None,
                        },
                    ))
                })
                .sorted_by_key(|(_, n)| n.syntax().text_range().start());
            let mut res = String::new();
            for (def, name) in defs {
                let variances = db.variances_of(def);
                if variances.is_empty() {
                    continue;
                }
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
                        .zip_eq(variances)
                        .format_with(", ", |(name, var), f| f(&format_args!(
                            "{}: {}",
                            name.as_str(),
                            match var {
                                Variance::Covariant => "covariant",
                                Variance::Invariant => "invariant",
                                Variance::Contravariant => "contravariant",
                                Variance::Bivariant => "bivariant",
                            },
                        )))
                );
            }

            expected.assert_eq(&res);
        })
    }
}
