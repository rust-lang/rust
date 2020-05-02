#![warn(clippy::needless_lifetimes)]
#![allow(dead_code, clippy::needless_pass_by_value)]

fn distinct_lifetimes<'a, 'b>(_x: &'a u8, _y: &'b u8, _z: u8) {}

fn distinct_and_static<'a, 'b>(_x: &'a u8, _y: &'b u8, _z: &'static u8) {}

// No error; same lifetime on two params.
fn same_lifetime_on_input<'a>(_x: &'a u8, _y: &'a u8) {}

// No error; static involved.
fn only_static_on_input(_x: &u8, _y: &u8, _z: &'static u8) {}

fn mut_and_static_input(_x: &mut u8, _y: &'static str) {}

fn in_and_out<'a>(x: &'a u8, _y: u8) -> &'a u8 {
    x
}

// No error; multiple input refs.
fn multiple_in_and_out_1<'a>(x: &'a u8, _y: &'a u8) -> &'a u8 {
    x
}

// No error; multiple input refs.
fn multiple_in_and_out_2<'a, 'b>(x: &'a u8, _y: &'b u8) -> &'a u8 {
    x
}

// No error; static involved.
fn in_static_and_out<'a>(x: &'a u8, _y: &'static u8) -> &'a u8 {
    x
}

// No error.
fn deep_reference_1<'a, 'b>(x: &'a u8, _y: &'b u8) -> Result<&'a u8, ()> {
    Ok(x)
}

// No error; two input refs.
fn deep_reference_2<'a>(x: Result<&'a u8, &'a u8>) -> &'a u8 {
    x.unwrap()
}

fn deep_reference_3<'a>(x: &'a u8, _y: u8) -> Result<&'a u8, ()> {
    Ok(x)
}

// Where-clause, but without lifetimes.
fn where_clause_without_lt<'a, T>(x: &'a u8, _y: u8) -> Result<&'a u8, ()>
where
    T: Copy,
{
    Ok(x)
}

type Ref<'r> = &'r u8;

// No error; same lifetime on two params.
fn lifetime_param_1<'a>(_x: Ref<'a>, _y: &'a u8) {}

fn lifetime_param_2<'a, 'b>(_x: Ref<'a>, _y: &'b u8) {}

// No error; bounded lifetime.
fn lifetime_param_3<'a, 'b: 'a>(_x: Ref<'a>, _y: &'b u8) {}

// No error; bounded lifetime.
fn lifetime_param_4<'a, 'b>(_x: Ref<'a>, _y: &'b u8)
where
    'b: 'a,
{
}

struct Lt<'a, I: 'static> {
    x: &'a I,
}

// No error; fn bound references `'a`.
fn fn_bound<'a, F, I>(_m: Lt<'a, I>, _f: F) -> Lt<'a, I>
where
    F: Fn(Lt<'a, I>) -> Lt<'a, I>,
{
    unreachable!()
}

fn fn_bound_2<'a, F, I>(_m: Lt<'a, I>, _f: F) -> Lt<'a, I>
where
    for<'x> F: Fn(Lt<'x, I>) -> Lt<'x, I>,
{
    unreachable!()
}

// No error; see below.
fn fn_bound_3<'a, F: FnOnce(&'a i32)>(x: &'a i32, f: F) {
    f(x);
}

fn fn_bound_3_cannot_elide() {
    let x = 42;
    let p = &x;
    let mut q = &x;
    // This will fail if we elide lifetimes of `fn_bound_3`.
    fn_bound_3(p, |y| q = y);
}

// No error; multiple input refs.
fn fn_bound_4<'a, F: FnOnce() -> &'a ()>(cond: bool, x: &'a (), f: F) -> &'a () {
    if cond {
        x
    } else {
        f()
    }
}

struct X {
    x: u8,
}

impl X {
    fn self_and_out<'s>(&'s self) -> &'s u8 {
        &self.x
    }

    // No error; multiple input refs.
    fn self_and_in_out<'s, 't>(&'s self, _x: &'t u8) -> &'s u8 {
        &self.x
    }

    fn distinct_self_and_in<'s, 't>(&'s self, _x: &'t u8) {}

    // No error; same lifetimes on two params.
    fn self_and_same_in<'s>(&'s self, _x: &'s u8) {}
}

struct Foo<'a>(&'a u8);

impl<'a> Foo<'a> {
    // No error; lifetime `'a` not defined in method.
    fn self_shared_lifetime(&self, _: &'a u8) {}
    // No error; bounds exist.
    fn self_bound_lifetime<'b: 'a>(&self, _: &'b u8) {}
}

fn already_elided<'a>(_: &u8, _: &'a u8) -> &'a u8 {
    unimplemented!()
}

fn struct_with_lt<'a>(_foo: Foo<'a>) -> &'a str {
    unimplemented!()
}

// No warning; two input lifetimes (named on the reference, anonymous on `Foo`).
fn struct_with_lt2<'a>(_foo: &'a Foo) -> &'a str {
    unimplemented!()
}

// No warning; two input lifetimes (anonymous on the reference, named on `Foo`).
fn struct_with_lt3<'a>(_foo: &Foo<'a>) -> &'a str {
    unimplemented!()
}

// No warning; two input lifetimes.
fn struct_with_lt4<'a, 'b>(_foo: &'a Foo<'b>) -> &'a str {
    unimplemented!()
}

trait WithLifetime<'a> {}

type WithLifetimeAlias<'a> = dyn WithLifetime<'a>;

// Should not warn because it won't build without the lifetime.
fn trait_obj_elided<'a>(_arg: &'a dyn WithLifetime) -> &'a str {
    unimplemented!()
}

// Should warn because there is no lifetime on `Drop`, so this would be
// unambiguous if we elided the lifetime.
fn trait_obj_elided2<'a>(_arg: &'a dyn Drop) -> &'a str {
    unimplemented!()
}

type FooAlias<'a> = Foo<'a>;

fn alias_with_lt<'a>(_foo: FooAlias<'a>) -> &'a str {
    unimplemented!()
}

// No warning; two input lifetimes (named on the reference, anonymous on `FooAlias`).
fn alias_with_lt2<'a>(_foo: &'a FooAlias) -> &'a str {
    unimplemented!()
}

// No warning; two input lifetimes (anonymous on the reference, named on `FooAlias`).
fn alias_with_lt3<'a>(_foo: &FooAlias<'a>) -> &'a str {
    unimplemented!()
}

// No warning; two input lifetimes.
fn alias_with_lt4<'a, 'b>(_foo: &'a FooAlias<'b>) -> &'a str {
    unimplemented!()
}

fn named_input_elided_output<'a>(_arg: &'a str) -> &str {
    unimplemented!()
}

fn elided_input_named_output<'a>(_arg: &str) -> &'a str {
    unimplemented!()
}

fn trait_bound_ok<'a, T: WithLifetime<'static>>(_: &'a u8, _: T) {
    unimplemented!()
}
fn trait_bound<'a, T: WithLifetime<'a>>(_: &'a u8, _: T) {
    unimplemented!()
}

// Don't warn on these; see issue #292.
fn trait_bound_bug<'a, T: WithLifetime<'a>>() {
    unimplemented!()
}

// See issue #740.
struct Test {
    vec: Vec<usize>,
}

impl Test {
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        unimplemented!()
    }
}

trait LintContext<'a> {}

fn f<'a, T: LintContext<'a>>(_: &T) {}

fn test<'a>(x: &'a [u8]) -> u8 {
    let y: &'a u8 = &x[5];
    *y
}

// Issue #3284: give hint regarding lifetime in return type.
struct Cow<'a> {
    x: &'a str,
}
fn out_return_type_lts<'a>(e: &'a str) -> Cow<'a> {
    unimplemented!()
}

// Make sure we still warn on implementations
mod issue4291 {
    trait BadTrait {
        fn needless_lt<'a>(x: &'a u8) {}
    }

    impl BadTrait for () {
        fn needless_lt<'a>(_x: &'a u8) {}
    }
}

fn main() {}
