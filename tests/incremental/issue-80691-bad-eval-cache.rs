//@ revisions: rfail1 rfail2
//@ failure-status: 101
//@ error-pattern: not implemented
//@ needs-unwind -Cpanic=abort causes abort instead of exit(101)

pub trait Interner {
    type InternedVariableKinds;
}

trait RustIrDatabase<I: Interner> {
    fn associated_ty_data(&self) -> AssociatedTyDatum<I>;
    fn impl_datum(&self) -> ImplDatum<I>;
}

trait Fold<I: Interner> {
    type Result;
}
impl<T, I: Interner> Fold<I> for Binders<T>
where
    T: HasInterner<Interner = I> + Fold<I>,
    <T as Fold<I>>::Result: HasInterner<Interner = I>,
    I: Interner,
{
    type Result = Binders<T::Result>;
}
impl<I: Interner> Fold<I> for WhereClause<I> {
    type Result = Binders<WhereClause<I>>;
}

trait HasInterner {
    type Interner: Interner;
}
impl<T: HasInterner> HasInterner for Vec<T> {
    type Interner = T::Interner;
}
impl<T: HasInterner + ?Sized> HasInterner for &T {
    type Interner = T::Interner;
}

pub struct VariableKind<I: Interner> {
    _marker: std::marker::PhantomData<I>,
}

struct VariableKinds<I: Interner> {
    _interned: I::InternedVariableKinds,
}

struct WhereClause<I: Interner> {
    _marker: std::marker::PhantomData<I>,
}
impl<I: Interner> HasInterner for WhereClause<I> {
    type Interner = I;
}

struct Binders<T> {
    _marker: std::marker::PhantomData<T>,
}
impl<T: HasInterner> HasInterner for Binders<T> {
    type Interner = T::Interner;
}
impl<T> Binders<&T> {
    fn cloned(self) -> Binders<T> {
        unimplemented!()
    }
}
impl<T: HasInterner> Binders<T> {
    fn map_ref<'a, U, OP>(&'a self, _op: OP) -> Binders<U>
    where
        OP: FnOnce(&'a T) -> U,
        U: HasInterner<Interner = T::Interner>,
    {
        unimplemented!()
    }
}
impl<T, I: Interner> Binders<T>
where
    T: Fold<I> + HasInterner<Interner = I>,
    I: Interner,
{
    fn substitute(self) -> T::Result {
        unimplemented!()
    }
}
impl<V, U> IntoIterator for Binders<V>
where
    V: HasInterner + IntoIterator<Item = U>,
    U: HasInterner<Interner = V::Interner>,
{
    type Item = Binders<U>;
    type IntoIter = BindersIntoIterator<V>;
    fn into_iter(self) -> Self::IntoIter {
        unimplemented!()
    }
}
struct BindersIntoIterator<V: HasInterner> {
    _binders: VariableKinds<V::Interner>,
}
impl<V> Iterator for BindersIntoIterator<V>
where
    V: HasInterner + IntoIterator,
    <V as IntoIterator>::Item: HasInterner<Interner = V::Interner>,
{
    type Item = Binders<<V as IntoIterator>::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}

struct ImplDatum<I: Interner> {
    binders: Binders<ImplDatumBound<I>>,
}
struct ImplDatumBound<I: Interner> {
    where_clauses: Vec<Binders<WhereClause<I>>>,
}
impl<I: Interner> HasInterner for ImplDatumBound<I> {
    type Interner = I;
}

struct AssociatedTyDatum<I: Interner> {
    binders: Binders<AssociatedTyDatumBound<I>>,
}

struct AssociatedTyDatumBound<I: Interner> {
    where_clauses: Vec<Binders<WhereClause<I>>>,
}
impl<I: Interner> HasInterner for AssociatedTyDatumBound<I> {
    type Interner = I;
}

struct ClauseBuilder<'me, I: Interner> {
    db: &'me dyn RustIrDatabase<I>,
}
impl<'me, I: Interner> ClauseBuilder<'me, I> {
    fn new() -> Self {
        unimplemented!()
    }
    fn push_clause(&mut self, _conditions: impl Iterator<Item = Binders<Binders<WhereClause<I>>>>) {
        unimplemented!()
    }
}

pub(crate) struct Forest<I: Interner> {
    _marker: std::marker::PhantomData<I>,
}

impl<I: Interner> Forest<I> {
    fn iter_answers<'f>(&'f self) {
        let builder = &mut ClauseBuilder::<I>::new();
        let impl_datum = builder.db.impl_datum();
        let impl_where_clauses = impl_datum
            .binders
            .map_ref(|b| &b.where_clauses)
            .into_iter()
            .map(|wc| wc.cloned().substitute());
        let associated_ty = builder.db.associated_ty_data();
        let assoc_ty_where_clauses = associated_ty
            .binders
            .map_ref(|b| &b.where_clauses)
            .into_iter()
            .map(|wc| wc.cloned().substitute());
        builder.push_clause(impl_where_clauses.chain(assoc_ty_where_clauses));
    }
}

pub struct SLGSolver {
    pub(crate) forest: Forest<ChalkIr>,
}
impl SLGSolver {
    fn new() -> Self {
        unimplemented!()
    }
    fn solve_multiple(&self) {
        let _answers = self.forest.iter_answers();
    }
}

pub struct ChalkIr;
impl Interner for ChalkIr {
    type InternedVariableKinds = Vec<VariableKind<ChalkIr>>;
}

fn main() {
    let solver = SLGSolver::new();
    solver.solve_multiple();
}
