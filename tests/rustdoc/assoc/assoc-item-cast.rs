#![crate_name = "foo"]

pub trait Expression {
    type SqlType;
}

pub trait AsExpression<T> {
    type Expression: Expression<SqlType = T>;
    fn as_expression(self) -> Self::Expression;
}

//@ has foo/type.AsExprOf.html
//@ has - '//pre[@class="rust item-decl"]' 'type AsExprOf<Item, Type> = <Item as AsExpression<Type>>::Expression;'
pub type AsExprOf<Item, Type> = <Item as AsExpression<Type>>::Expression;
