//@ check-pass
//@ known-bug: #84533

// Should fail. Lifetimes are checked correctly when `foo` is called, but NOT
// when only the lifetime parameters are instantiated.

use std::marker::PhantomData;

#[allow(dead_code)]
fn foo<'b, 'a>() -> PhantomData<&'b &'a ()> {
    PhantomData
}

#[allow(dead_code)]
#[allow(path_statements)]
fn caller<'b, 'a>() {
    foo::<'b, 'a>;
}

// In contrast to above, below code correctly does NOT compile.
// fn caller<'b, 'a>() {
//     foo::<'b, 'a>();
// }

// error: lifetime may not live long enough
//   --> src/main.rs:22:5
//   |
// 21 | fn caller<'b, 'a>() {
//   |           --  -- lifetime `'a` defined here
//   |           |
//   |           lifetime `'b` defined here
// 22 |     foo::<'b, 'a>();
//   |     ^^^^^^^^^^^^^^^ requires that `'a` must outlive `'b`
//   |
//   = help: consider adding the following bound: `'a: 'b`

fn main() {}
