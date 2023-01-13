#![feature(auto_traits)]
#![feature(negative_impls)]

pub trait PubPrincipal {}
auto trait PrivNonPrincipal {}

pub fn leak_dyn_nonprincipal() -> Box<dyn PubPrincipal + PrivNonPrincipal> { loop {} }
//~^ WARN private trait `PrivNonPrincipal` in public interface
//~| WARN this was previously accepted

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
