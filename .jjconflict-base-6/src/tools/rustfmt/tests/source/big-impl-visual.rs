// rustfmt-indent_style: Visual

// #1357
impl<
    'a,
    Select,
    From,
    Distinct,
    Where,
    Order,
    Limit,
    Offset,
    Groupby,
    DB,
> InternalBoxedDsl<'a, DB>
    for SelectStatement<
        Select,
        From,
        Distinct,
        Where,
        Order,
        Limit,
        Offset,
        GroupBy,
    > where
        DB: Backend,
        Select: QueryFragment<DB> + SelectableExpression<From> + 'a,
        Distinct: QueryFragment<DB> + 'a,
        Where: Into<Option<Box<QueryFragment<DB> + 'a>>>,
        Order: QueryFragment<DB> + 'a,
        Limit: QueryFragment<DB> + 'a,
        Offset: QueryFragment<DB> + 'a,
{
    type Output = BoxedSelectStatement<'a, Select::SqlTypeForSelect, From, DB>;

    fn internal_into_boxed(self) -> Self::Output {
        BoxedSelectStatement::new(
            Box::new(self.select),
            self.from,
            Box::new(self.distinct),
            self.where_clause.into(),
            Box::new(self.order),
            Box::new(self.limit),
            Box::new(self.offset),
        )
    }
}

// #1369
impl<
    ExcessivelyLongGenericName,
      ExcessivelyLongGenericName,
    AnotherExcessivelyLongGenericName,
> Foo for Bar {
    fn foo() {}
}
impl Foo<
    ExcessivelyLongGenericName,
      ExcessivelyLongGenericName,
    AnotherExcessivelyLongGenericName,
> for Bar {
    fn foo() {}
}
impl<
    ExcessivelyLongGenericName,
    ExcessivelyLongGenericName,
    AnotherExcessivelyLongGenericName,
> Foo<
    ExcessivelyLongGenericName,
      ExcessivelyLongGenericName,
    AnotherExcessivelyLongGenericName,
> for Bar {
    fn foo() {}
}
impl<
    ExcessivelyLongGenericName,
      ExcessivelyLongGenericName,
    AnotherExcessivelyLongGenericName,
> Foo for Bar<
    ExcessivelyLongGenericName,
    ExcessivelyLongGenericName,
    AnotherExcessivelyLongGenericName,
> {
    fn foo() {}
}
impl Foo<
    ExcessivelyLongGenericName,
      ExcessivelyLongGenericName,
    AnotherExcessivelyLongGenericName,
> for Bar<
    ExcessivelyLongGenericName,
    ExcessivelyLongGenericName,
    AnotherExcessivelyLongGenericName,
> {
    fn foo() {}
}
impl<ExcessivelyLongGenericName,
     ExcessivelyLongGenericName,
     AnotherExcessivelyLongGenericName> Foo<ExcessivelyLongGenericName,
                                            ExcessivelyLongGenericName,
                                            AnotherExcessivelyLongGenericName>
    for Bar<ExcessivelyLongGenericName,
            ExcessivelyLongGenericName,
            AnotherExcessivelyLongGenericName> {
    fn foo() {}
}
