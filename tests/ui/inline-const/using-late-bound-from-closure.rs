// Test for ICE: cannot convert ReLateParam to a region vid
// https://github.com/rust-lang/rust/issues/125873

#![feature(closure_lifetime_binder)]
fn foo() {
    let a = for<'a> |b: &'a ()| -> &'a () {
        const {
            let awd = ();
            let _: &'a () = &awd;
            //~^ ERROR `awd` does not live long enough
        };
        b
    };
}

fn main() {}
