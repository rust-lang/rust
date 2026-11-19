// Regression test for <https://github.com/rust-lang/rust/issues/141504>.
// A closure with an explicit `for<...>` binder is higher-ranked over its own
// lifetimes, even when the expected signature is less general.

#![feature(closure_lifetime_binder)]

fn takes_static(_: impl FnOnce(&'static u8)) {}
fn callback(_: impl for<'a> FnOnce(&'a mut i32)) {}
fn callback_no_args(_: impl FnOnce()) {}
fn two_args(_: impl Fn(&u8, &u8)) {}

fn expected_static() {
    takes_static(for<'a> |x: &'a u8| -> () {
        let _: &'static u8 = x;
        //~^ ERROR lifetime may not live long enough
    });
}

fn expected_higher_ranked() {
    callback(for<'a> |y: &'a mut i32| -> () {
        let _: &'static mut i32 = y;
        //~^ ERROR lifetime may not live long enough
    });
}

fn outer_lifetime<'x>(r: &'x i32) {
    callback_no_args(for<'a> || -> () {
        let _: &'a i32 = r;
        //~^ ERROR lifetime may not live long enough
    });
}

fn wrong_arg_count() {
    // The expected signature is still checked for the argument count.
    two_args(for<'a> |_: &'a u8| -> () {});
    //~^ ERROR closure is expected to take 2 arguments, but it takes 1 argument
}

fn main() {}
