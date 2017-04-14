// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
