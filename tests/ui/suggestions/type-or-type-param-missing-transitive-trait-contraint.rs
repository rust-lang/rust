//@ run-rustfix
#![allow(warnings)]
use std::collections::HashMap;

#[derive(Clone)]
struct Ctx<A> {
    a_map: HashMap<String, B<A>>,
}

#[derive(Clone)]
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

fn main() {}
