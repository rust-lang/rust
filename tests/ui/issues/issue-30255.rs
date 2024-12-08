//
// Test that lifetime elision error messages correctly omit parameters
// with no elided lifetimes

struct S<'a> {
    field: &'a i32,
}

fn f(a: &S, b: i32) -> &i32 {
//~^ ERROR missing lifetime specifier [E0106]
    panic!();
}

fn g(a: &S, b: bool, c: &i32) -> &i32 {
//~^ ERROR missing lifetime specifier [E0106]
    panic!();
}

fn h(a: &bool, b: bool, c: &S, d: &i32) -> &i32 {
//~^ ERROR missing lifetime specifier [E0106]
    panic!();
}

fn main() {}
