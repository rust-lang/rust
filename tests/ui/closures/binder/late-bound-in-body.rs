//@ check-pass

#![feature(closure_lifetime_binder)]

fn main() {
    let _ = for<'a> || -> () {
        let _: &'a bool = &true;
    };
}
