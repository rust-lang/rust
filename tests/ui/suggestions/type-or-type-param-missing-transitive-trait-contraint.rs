//@ run-rustfix
#![allow(warnings)]
use std::collections::HashMap;

#[derive(Clone)]
struct Ctx<A> {
    a_map: HashMap<String, B<A>>,
}

#[derive(Clone, PartialEq, Eq)]
struct B<A> {
    a: A,
}

fn foo<Z>(ctx: &mut Ctx<Z>) {
    let a_map = ctx.a_map.clone(); //~ ERROR E0599
}

struct S;
fn bar(ctx: &mut Ctx<S>) {
    let a_map = ctx.a_map.clone(); //~ ERROR E0599
}

fn qux<Z>(ctx: &mut Ctx<Z>) {
    ctx.a_map["a"].eq(&ctx.a_map["a"]); //~ ERROR E0599
    <_ as Eq>::assert_receiver_is_total_eq(&ctx.a_map["a"]); //~ ERROR E0277
}

fn qut(ctx: &mut Ctx<S>) {
    ctx.a_map["a"].eq(&ctx.a_map["a"]); //~ ERROR E0599
    <_ as Eq>::assert_receiver_is_total_eq(&ctx.a_map["a"]); //~ ERROR E0277
}

fn main() {}
