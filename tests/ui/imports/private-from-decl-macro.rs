//@ revisions: pass fail
//@[pass] check-pass

#![feature(decl_macro)]

mod m {
    // Name in two namespaces, one public, one private.
    // The private name is filtered away when importing, even from a macro 2.0
    pub struct S {}
    const S: u8 = 0;

    pub macro mac_single($S:ident) {
        use crate::m::$S;
    }

    pub macro mac_glob() {
        use crate::m::*;
    }
}

mod single {
    crate::m::mac_single!(S);

    fn check() {
        let s = S {};
        #[cfg(fail)]
        let s = S; //[fail]~ ERROR expected value, found struct `S`
    }
}

mod glob {
    crate::m::mac_glob!();
}

fn main() {}
