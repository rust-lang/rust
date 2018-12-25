#![crate_name = "foo"]

// ignore-tidy-linelength

pub trait Expression {
    type SqlType;
}

pub trait AsExpression<T> {
    type Expression: Expression<SqlType = T>;
    fn as_expression(self) -> Self::Expression;
}

// @has foo/type.AsExprOf.html
// @has - '//*[@class="rust typedef"]' 'type AsExprOf<Item, Type> = <Item as AsExpression<Type>>::Expression;'
pub type AsExprOf<Item, Type> = <Item as AsExpression<Type>>::Expression;
