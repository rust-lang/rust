fn main() {
    for _ in [1, 2, 3].iter()map(|x| x) {}
    //~^ ERROR expected `{`, found `map`
    //~| HELP you might have meant to write a method call
}
fn foo1() {
    for _ in 1.3f64 cos() {}
    //~^ ERROR expected `{`, found `cos`
    //~| HELP you might have meant to write a method call
}
fn foo2() {
    for _ in 1.3 cos {}
    //~^ ERROR expected `{`, found `cos`
    //~| HELP you might have meant to write a field access
}
fn foo3() {
    for _ in 1 cos() {}
    //~^ ERROR expected `{`, found `cos`
    //~| HELP you might have meant to write a method call
}
fn foo4() {
    for _ in 1 cos {}
    //~^ ERROR expected `{`, found `cos`
    //~| HELP you might have meant to write a field access
}
fn foo5() {
    let x = (vec![1, 2, 3],);
    for _ in x 0 {}
    //~^ ERROR expected `{`, found `0`
    //~| HELP you might have meant to write a field access
}
fn foo6() {
    let x = ((vec![1, 2, 3],),);
    for _ in x 0.0 {}
    //~^ ERROR expected `{`, found `0.0`
    //~| HELP you might have meant to write a field access
}
fn foo7() {
    let x = Some(vec![1, 2, 3]);
    for _ in x unwrap() {}
    //~^ ERROR expected `{`, found `unwrap`
    //~| HELP you might have meant to write a method call
}
fn foo8() {
    let x = S { a: A { b: vec![1, 2, 3] } };
    for _ in x a.b {}
    //~^ ERROR expected `{`, found `a`
    //~| HELP you might have meant to write a field access
}

struct S {
    a: A,
}

struct A {
    b: Vec<i32>,
}
