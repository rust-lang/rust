#![feature(auto_traits)]
#![feature(negative_impls)]
#![allow(non_local_definitions)]

pub trait PubPrincipal {}
auto trait PrivNonPrincipal {}

pub fn leak_dyn_nonprincipal() -> Box<dyn PubPrincipal + PrivNonPrincipal> { loop {} }
//~^ WARN trait `PrivNonPrincipal` is more private than the item `leak_dyn_nonprincipal`

#[deny(missing_docs)]
fn container() {
    impl dyn PubPrincipal {
        pub fn check_doc_lint() {} //~ ERROR missing documentation for an associated function
    }
    impl dyn PubPrincipal + PrivNonPrincipal {
        pub fn check_doc_lint() {} // OK, no missing doc lint
    }
}

fn main() {}
