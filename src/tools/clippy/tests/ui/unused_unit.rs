

// The output for humans should just highlight the whole span without showing
// the suggested replacement, but we also want to test that suggested
// replacement only removes one set of parentheses, rather than naïvely
// stripping away any starting or ending parenthesis characters—hence this
// test of the JSON error format.

#![feature(custom_inner_attributes)]
#![feature(closure_lifetime_binder)]
#![rustfmt::skip]

#![deny(clippy::unused_unit)]
#![allow(dead_code)]
#![allow(clippy::from_over_into)]

struct Unitter;
impl Unitter {
    #[allow(clippy::no_effect)]
    pub fn get_unit<F: Fn() -> (), G>(&self, f: F, _g: G) -> ()
    //~^ unused_unit
    //~| unused_unit
    where G: Fn() -> () {
    //~^ unused_unit
        let _y: &dyn Fn() -> () = &f;
        //~^ unused_unit
        (); // this should not lint, as it's not in return type position
    }
}

impl Into<()> for Unitter {
    #[rustfmt::skip]
    fn into(self) -> () {
    //~^ unused_unit
        ()
        //~^ unused_unit
    }
}

trait Trait {
    fn redundant<F: FnOnce() -> (), G, H>(&self, _f: F, _g: G, _h: H)
    //~^ unused_unit
    where
        G: FnMut() -> (),
        //~^ unused_unit
        H: Fn() -> ();
        //~^ unused_unit
}

impl Trait for Unitter {
    fn redundant<F: FnOnce() -> (), G, H>(&self, _f: F, _g: G, _h: H)
    //~^ unused_unit
    where
        G: FnMut() -> (),
        //~^ unused_unit
        H: Fn() -> () {}
        //~^ unused_unit
}

fn return_unit() -> () { () }
//~^ unused_unit
//~| unused_unit

#[allow(clippy::needless_return)]
#[allow(clippy::never_loop)]
#[allow(clippy::unit_cmp)]
fn main() {
    let u = Unitter;
    assert_eq!(u.get_unit(|| {}, return_unit), u.into());
    return_unit();
    loop {
        break();
        //~^ unused_unit
    }
    return();
    //~^ unused_unit
}

// https://github.com/rust-lang/rust-clippy/issues/4076
fn foo() {
    macro_rules! foo {
        (recv($r:expr) -> $res:pat => $body:expr) => {
            $body
        }
    }

    foo! {
        recv(rx) -> _x => ()
    }
}

#[rustfmt::skip]
fn test()->(){}
//~^ unused_unit

#[rustfmt::skip]
fn test2() ->(){}
//~^ unused_unit

#[rustfmt::skip]
fn test3()-> (){}
//~^ unused_unit

fn macro_expr() {
    macro_rules! e {
        () => (());
    }
    e!()
}

mod issue9748 {
    fn main() {
        let _ = for<'a> |_: &'a u32| -> () {};
    }
}

mod issue9949 {
    fn main() {
        #[doc = "documentation"]
        ()
    }
}
