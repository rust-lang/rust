fn reflow_list_node_with_rule(node: &CompoundNode, rule: &Rule, args: &[Arg], shape: &Shape)
where
    T: FOo,
    U: Bar,
{
    let mut effects = HashMap::new();
}

fn reflow_list_node_with_rule(node: &CompoundNode, rule: &Rule, args: &[Arg], shape: &Shape)
where
    T: FOo,
{
    let mut effects = HashMap::new();
}

fn reflow_list_node_with_rule(
    node: &CompoundNode,
    rule: &Rule,
    args: &[Arg],
    shape: &Shape,
    shape: &Shape,
) where
    T: FOo,
    U: Bar,
{
    let mut effects = HashMap::new();
}

fn reflow_list_node_with_rule(
    node: &CompoundNode,
    rule: &Rule,
    args: &[Arg],
    shape: &Shape,
    shape: &Shape,
) where
    T: FOo,
{
    let mut effects = HashMap::new();
}

fn reflow_list_node_with_rule(
    node: &CompoundNode,
    rule: &Rule,
    args: &[Arg],
    shape: &Shape,
) -> Option<String>
where
    T: FOo,
    U: Bar,
{
    let mut effects = HashMap::new();
}

fn reflow_list_node_with_rule(
    node: &CompoundNode,
    rule: &Rule,
    args: &[Arg],
    shape: &Shape,
) -> Option<String>
where
    T: FOo,
{
    let mut effects = HashMap::new();
}

pub trait Test {
    fn very_long_method_name<F>(self, f: F) -> MyVeryLongReturnType
    where
        F: FnMut(Self::Item) -> bool;

    fn exactly_100_chars1<F>(self, f: F) -> MyVeryLongReturnType
    where
        F: FnMut(Self::Item) -> bool;
}

fn very_long_function_name<F>(very_long_argument: F) -> MyVeryLongReturnType
where
    F: FnMut(Self::Item) -> bool,
{
}

struct VeryLongTupleStructName<A, B, C, D, E>(LongLongTypename, LongLongTypename, i32, i32)
where
    A: LongTrait;

struct Exactly100CharsToSemicolon<A, B, C, D, E>(LongLongTypename, i32, i32)
where
    A: LongTrait1234;

struct AlwaysOnNextLine<LongLongTypename, LongTypename, A, B, C, D, E, F>
where
    A: LongTrait,
{
    x: i32,
}

pub trait SomeTrait<T>
where
    T: Something
        + Sync
        + Send
        + Display
        + Debug
        + Copy
        + Hash
        + Debug
        + Display
        + Write
        + Read
        + FromStr,
{
}

// #2020
impl<'a, 'gcx, 'tcx> ProbeContext<'a, 'gcx, 'tcx> {
    fn elaborate_bounds<F>(&mut self, bounds: &[ty::PolyTraitRef<'tcx>], mut mk_cand: F)
    where
        F: for<'b> FnMut(
            &mut ProbeContext<'b, 'gcx, 'tcx>,
            ty::PolyTraitRef<'tcx>,
            ty::AssociatedItem,
        ),
    {
        // ...
    }
}

// #2497
fn handle_update<'a, Tab, Conn, R, C>(
    executor: &Executor<PooledConnection<ConnectionManager<Conn>>>,
    change_set: &'a C,
) -> ExecutionResult
where
    &'a C: Identifiable + AsChangeset<Target = Tab> + HasTable<Table = Tab>,
    <&'a C as AsChangeset>::Changeset: QueryFragment<Conn::Backend>,
    Tab: Table + HasTable<Table = Tab>,
    Tab::PrimaryKey: EqAll<<&'a C as Identifiable>::Id>,
    Tab::FromClause: QueryFragment<Conn::Backend>,
    Tab: FindDsl<<&'a C as Identifiable>::Id>,
    Find<Tab, <&'a C as Identifiable>::Id>: IntoUpdateTarget<Table = Tab>,
    <Find<Tab, <&'a C as Identifiable>::Id> as IntoUpdateTarget>::WhereClause:
        QueryFragment<Conn::Backend>,
    Tab::Query: FilterDsl<<Tab::PrimaryKey as EqAll<<&'a C as Identifiable>::Id>>::Output>,
    Filter<Tab::Query, <Tab::PrimaryKey as EqAll<<&'a C as Identifiable>::Id>>::Output>: LimitDsl,
    Limit<Filter<Tab::Query, <Tab::PrimaryKey as EqAll<<&'a C as Identifiable>::Id>>::Output>>:
        QueryDsl
            + BoxedDsl<
                'a,
                Conn::Backend,
                Output = BoxedSelectStatement<'a, R::SqlType, Tab, Conn::Backend>,
            >,
    R: LoadingHandler<Conn, Table = Tab, SqlType = Tab::SqlType>
        + GraphQLType<TypeInfo = (), Context = ()>,
{
    unimplemented!()
}
