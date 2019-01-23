// compile-pass

// The output for humans should just highlight the whole span without showing
// the suggested replacement, but we also want to test that suggested
// replacement only removes one set of parentheses, rather than naïvely
// stripping away any starting or ending parenthesis characters—hence this
// test of the JSON error format.

#![deny(clippy::unused_unit)]
#![allow(clippy::needless_return)]

struct Unitter;
impl Unitter {
    // try to disorient the lint with multiple unit returns and newlines
    pub fn get_unit<F: Fn() -> (), G>(&self, f: F, _g: G) ->
        ()
    where G: Fn() -> () {
        let _y: &Fn() -> () = &f;
        (); // this should not lint, as it's not in return type position
    }
}

impl Into<()> for Unitter {
    #[rustfmt::skip]
    fn into(self) -> () {
        ()
    }
}

fn return_unit() -> () { () }

fn main() {
    let u = Unitter;
    assert_eq!(u.get_unit(|| {}, return_unit), u.into());
    return_unit();
    loop {
        break();
    }
    return();
}
