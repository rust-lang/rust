// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use m1::m2::*;

pub mod m1 {
    pub mod m2 {
        pub fn who_am_i() -> ~str { (func!()).to_owned() }

        pub trait Tr {
            fn defme(&self) -> ~str { 
                (func!()).to_owned()
            }
            fn must_implement(&self) -> ~str;
        }

        pub struct St1;
        impl Tr for St1 {
            fn defme(&self) -> ~str { 
                (func!()).to_owned()
            }
            fn must_implement(&self) -> ~str {
                (func!()).to_owned()
            }
        }

        pub struct St2;
        impl Tr for St2 {
            // defme() should get named with Tr

            fn must_implement(&self) -> ~str {
                (func!()).to_owned()
            }
        }

        pub fn use_lambda() -> ~str {
            let lambda2 = || {
                "inside a lambda we get: %s" + func!()
            };
            lambda2()
        }

    } // end m2

} // end m1

//
//  tests for func!() macro : needs to handle all
//  the different places where functions can be
//  defined, include trait default methods, 
//  implementations, and lambdas.
//
pub fn main() {

    printfln!("m1::m2::who_am_i() is '%s'", m1::m2::who_am_i()); // m1::m2::__extensions__

    // The Windows tests are wrapped in an extra module for some reason.
    // And now we include a file location prefix to track lambdas easily, so
    // just use the ends_with() for all verifies:
    assert!(m1::m2::who_am_i().ends_with("m1::m2::who_am_i")); // OK.

    // with default methods in the trait Tr

    // overridden default method, overridden in St1
    let s1 = m1::m2::St1;
    printfln!("s1.defme() is '%s'", s1.defme()); // m1::m2::__extensions__ hmmm... not what we want.
//    assert!(s1.defme().ends_with("m1::m2::St1::defme"));

    // not overridden, using the default from the Trait Tr
    // Tr::defme should get named and used, since St2 didn't override it.
    let s2 = m1::m2::St2;
    printfln!("s2.defme() is '%s'", s2.defme()); // m1::m2::Tr hmm... want to get defme on there.
//    assert!(s2.defme().ends_with("m1::m2::Tr::defme"));


    // no default implementatio in the trait Tr

    let s1impl = s1.must_implement();
//    assert!((s1impl.ends_with("m1::m2::St1::must_implement")));
    // no default, in the extensions for St1
    let s2impl = s1.must_implement();
//    assert!((s2impl.ends_with("m1::m2::St2::must_implement")));

    // lambda. Lambdas are just called 'lambda' everywhere, but they
    // should retain module and enclosing function names.
    
    // local lambda
    let lambda = || {
        "inside a lambda we get: " + func!()
    };
    let l1 = lambda();
    printfln!("l1 is '%s'", l1); // 
//    assert!(l1.ends_with("lambda"));

    // lambda in module, in function
    let l2 = m1::m2::use_lambda();
    printfln!("l2 is '%s'", l2); // m1::m2::use_lambda. FAILs to add ::lambda at the moment.
//    assert!(l2.ends_with("m1::m2::use_lambda::lambda"));

    // local function wrapper of lambda
    fn use3() -> ~str {
        let lambda3 = || {
            "inside lambda3 we get: " + func!()
        };
        lambda3()
    }
          
    let l3 = use3();
    printfln!("l3 is '%s'", l3); // main::use3
//    assert!(l3.ends_with("use3::lambda"));
}

