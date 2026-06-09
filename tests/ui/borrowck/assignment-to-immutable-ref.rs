//! Regression test for issue <https://github.com/rust-lang/rust/issues/51515>
//! Test that assigning through an immutable reference (`&`) correctly yields
//! an assignment error (E0594) and suggests using a mutable reference.

fn main() {
    let foo = &16;
    //~^ HELP consider changing this to be a mutable reference
    *foo = 32;
    //~^ ERROR cannot assign to `*foo`, which is behind a `&` reference
    let bar = foo;
    //~^ HELP consider specifying this binding's type
    *bar = 64;
    //~^ ERROR cannot assign to `*bar`, which is behind a `&` reference
}
