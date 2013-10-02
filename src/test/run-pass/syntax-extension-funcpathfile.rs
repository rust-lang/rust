// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// test for the funcpathfile!() macro

use m1::m2::*;

pub mod m1 {
    pub mod m2 {
        pub fn who_am_i() -> ~str { (funcpathfile!()).to_owned() }

        pub trait Tr {
            fn defme(&self) -> ~str { 
                (funcpathfile!()).to_owned()
            }
            fn must_implement(&self) -> ~str;
        }

        pub struct St1;
        impl Tr for St1 {
            fn defme(&self) -> ~str { 
                (funcpathfile!()).to_owned()
            }
            fn must_implement(&self) -> ~str {
                (funcpathfile!()).to_owned()
            }
        }

        pub struct St2;
        impl Tr for St2 {
            // defme() should get named with Tr

            fn must_implement(&self) -> ~str {
                (funcpathfile!()).to_owned()
            }
        }

        pub fn use_lambda() -> ~str {
            let lambda2 = || {
                "inside a lambda we get: " + funcpathfile!()
            };
            lambda2()
        }

    } // end m2

} // end m1

//
//  tests for funcpathfile!() macro : needs to handle all
//  the different places where functions can be
//  defined, include trait default methods, 
//  implementations, and lambdas.
//
pub fn main() {

    info2!("m1::m2::who_am_i() is '{}'", m1::m2::who_am_i());

    // The Windows tests are wrapped in an extra module for some reason.
    // And in funcpathfile!() we include a file location prefix to track 
    // lambdas easily, so just use the ends_with() for all verifies:
    assert!(m1::m2::who_am_i().ends_with("m1::m2::who_am_i"));

    // with default methods in the trait Tr

    // overridden default method, overridden in St1
    let s1 = m1::m2::St1;
    info2!("s1.defme() is '{}'", s1.defme());
    assert!(s1.defme().ends_with("m1::m2::Tr$St1::defme"));

    // not overridden, using the default from the Trait Tr
    // Tr::defme should get named and used, since St2 didn't override it.
    let s2 = m1::m2::St2;
    info2!("s2.defme() is '{}'", s2.defme()); // m1::m2::Tr hmm... want to get defme on there.
    assert!(s2.defme().ends_with("m1::m2::Tr::defme"));


    // no default implementatio in the trait Tr

    let s1impl = s1.must_implement();
    info2!("s1impl is '{}'", s1impl);
    assert!((s1impl.ends_with("m1::m2::Tr$St1::must_implement")));
    // no default, in the extensions for St1
    let s2impl = s2.must_implement();
    info2!("s2impl is '{}'", s2impl);
    assert!((s2impl.ends_with("m1::m2::Tr$St2::must_implement")));

    // lambda. Lambdas are just called 'lambda' everywhere, but they
    // should retain module and enclosing function names.
    
    // local lambda
    let lambda = || {
        "inside a lambda we get: " + funcpathfile!()
    };
    let l1 = lambda();
    info2!("l1 is '{}'", l1);
    assert!(l1.ends_with("main::lambda"));

    // lambda in module, in function
    let l2 = m1::m2::use_lambda();
    info2!("l2 is '{}'", l2);
    assert!(l2.ends_with("m1::m2::use_lambda::lambda"));

    // distinguish a::mZ::c from a::mQ::c, where mZ and mQ are modules
    // This is why the implementation of funcpathfile!() must track module 
    // namespaces as well.  Getting back a::c just won't do.
    
    fn a() -> (&str, &str) {
        pub mod mZ {
            pub fn c() -> &str { funcpathfile!() }
        }
        pub mod mQ {
            pub fn c() -> &str { funcpathfile!() }
        }
        (mZ::c(), mQ::c())
    }

    let (cz, cq) = a();
    info2!("cz is {}", cz);
    info2!("cq is {}", cq);
    assert!(cz.ends_with("main::a::mZ::c"));
    assert!(cq.ends_with("main::a::mQ::c"));

    // and the symmetric case, where mod are duplicated:
    // If we got back only mod_path!() == "outer::M" and funcpathfile!() == "c"
    //  we would not know whether outer::a::M::c was called,
    //  or outer::b::M::c was called.
    mod outer {
        pub fn a() -> &str {
            pub mod M {
                pub fn c() -> &str { funcpathfile!() }
            }
            M::c()
        }
        pub fn b() -> &str {
            pub mod M {
                pub fn c() -> &str { funcpathfile!() }
            }
            M::c()
        }
    }

    let oamc = outer::a();
    let obmc = outer::b();
    info2!("oamc is {}", oamc);
    info2!("obmc is {}", obmc);
    assert!(oamc.ends_with("main::outer::a::M::c"));
    assert!(obmc.ends_with("main::outer::b::M::c"));



    // local function wrapper of lambda
    fn use3() -> ~str {
        let lambda3 = || {
            "inside lambda3 we get: " + funcpathfile!()
        };
        lambda3()
    }
          
    let l3 = use3();
    info2!("l3 is '{}'", l3); // main::use3
    assert!(l3.ends_with("main::use3::lambda"));

    // and named functions inside lambdas:
    let lambda4 = || {
        fn embedded() -> ~str {
            "inside lambda4::embedded we get: " + funcpathfile!()
        }
        embedded()
    };
    let l4 = lambda4();
    info2!("l4 is '{}'", l4);
    assert!(l4.ends_with("main::lambda::embedded"));

    // column information is needed to distinguish between multiple lambda
    // on a single line.
    let l5 = || { funcpathfile!() }; let l6 = || { funcpathfile!() };
    let s5 = l5();
    let s6 = l6();
    info2!("l5 is '{}', l6 is '{}'", s5, s6);
    assert!(s5 != s6);

}

