pub struct Nullable<T: NotNull>(T);

pub trait NotNull {}

pub trait IntoNullable {
    type Nullable;
}

impl<T: NotNull> IntoNullable for T {
    type Nullable = Nullable<T>;
}

impl<T: NotNull> IntoNullable for Nullable<T> {
    type Nullable = Nullable<T>;
}

pub trait Expression {
    type SqlType;
}

pub trait Column: Expression {}

#[derive(Debug, Copy, Clone)]
//~^ ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
pub enum ColumnInsertValue<Col, Expr> where
//~^ ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
//~| ERROR trait `NotNull` is not implemented for `<Col as Expression>::SqlType`
    Col: Column,
    Expr: Expression<SqlType=<Col::SqlType as IntoNullable>::Nullable>,
{
    Expression(Col, Expr),
    Default(Col),
}

fn main() {}
