// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// test for the function!() macro

use m1::m2::*;

pub mod m1 {
    pub mod m2 {
        pub fn who_am_i() -> ~str { (function!()).to_owned() }

        pub trait Tr {
            fn defme(&self) -> ~str { 
                (function!()).to_owned()
            }
            fn must_implement(&self) -> ~str;
        }

        pub struct St1;
        impl Tr for St1 {
            fn defme(&self) -> ~str { 
                (function!()).to_owned()
            }
            fn must_implement(&self) -> ~str {
                (function!()).to_owned()
            }
        }

        pub struct St2;
        impl Tr for St2 {
            // defme() should get named with Tr

            fn must_implement(&self) -> ~str {
                (function!()).to_owned()
            }
        }

        pub fn use_lambda() -> &str {
            let lambda2 = || {
                function!()
            };
            lambda2()
        }

    } // end m2

} // end m1

//
//  tests for function!() macro : needs to handle all
//  the different places where functions can be
//  defined, include trait default methods, 
//  implementations, and lambdas.
//
pub fn main() {

    info2!("m1::m2::who_am_i() is '{}'", m1::m2::who_am_i());

    // The Windows tests are wrapped in an extra module for some reason.
    // And in function!() we include a file location prefix to track 
    // lambdas easily, so just use the ends_with() for all verifies:
    assert_eq!(m1::m2::who_am_i(), ~"who_am_i");

    // with default methods in the trait Tr

    // overridden default method, overridden in St1
    let s1 = m1::m2::St1;
    info2!("s1.defme() is '{}'", s1.defme());
    assert_eq!(s1.defme(), ~"defme");

    // not overridden, using the default from the Trait Tr
    // Tr::defme should get named and used, since St2 didn't override it.
    let s2 = m1::m2::St2;
    info2!("s2.defme() is '{}'", s2.defme()); // m1::m2::Tr hmm... want to get defme on there.
    assert_eq!(s2.defme(), ~"defme");


    // no default implementatio in the trait Tr

    let s1impl = s1.must_implement();
    info2!("s1impl is '{}'", s1impl);
    assert_eq!(s1impl, ~"must_implement");
    // no default, in the extensions for St1
    let s2impl = s2.must_implement();
    info2!("s2impl is '{}'", s2impl);
    assert_eq!(s2impl, ~"must_implement");

    // lambda. Lambdas are just called 'lambda' everywhere, but they
    // should retain module and enclosing function names.
    
    // local lambda
    let lambda = || {
        function!()
    };
    let l1 = lambda();
    info2!("l1 is '{}'", l1);
    assert_eq!(l1, "lambda");

    // lambda in module, in function
    let l2 = m1::m2::use_lambda();
    info2!("l2 is '{}'", l2);
    assert_eq!(l2, "lambda");

    // distinguish a::mZ::c from a::mQ::c, where mZ and mQ are modules
    // This is why the implementation of function!() must track module 
    // namespaces as well.  Getting back a::c just won't do.
    
    fn a() -> (&str, &str) {
        pub mod mZ {
            pub fn c() -> &str { function!() }
        }
        pub mod mQ {
            pub fn c() -> &str { function!() }
        }
        (mZ::c(), mQ::c())
    }

    let (cz, cq) = a();
    info2!("cz is {}", cz);
    info2!("cq is {}", cq);
    assert_eq!(cz, "c");
    assert_eq!(cq, "c");

    // and the symmetric case, where mod are duplicated:
    // If we got back only mod_path!() == "outer::M" and function!() == "c"
    //  we would not know whether outer::a::M::c was called,
    //  or outer::b::M::c was called.
    mod outer {
        pub fn a() -> &str {
            pub mod M {
                pub fn c() -> &str { function!() }
            }
            M::c()
        }
        pub fn b() -> &str {
            pub mod M {
                pub fn c() -> &str { function!() }
            }
            M::c()
        }
    }

    let oamc = outer::a();
    let obmc = outer::b();
    info2!("oamc is {}", oamc);
    info2!("obmc is {}", obmc);
    assert_eq!(oamc, "c");
    assert_eq!(obmc, "c");



    // local function wrapper of lambda
    fn use3() -> &str {
        let lambda3 = || {
            function!()
        };
        lambda3()
    }
          
    let l3 = use3();
    info2!("l3 is '{}'", l3); // main::use3
    assert_eq!(l3, "lambda");

    // and named functions inside lambdas:
    let lambda4 = || {
        fn embedded() -> &str {
            function!()
        }
        embedded()
    };
    let l4 = lambda4();
    info2!("l4 is '{}'", l4);
    assert_eq!(l4, "embedded");

}

