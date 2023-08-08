#![feature(auto_traits)]
#![feature(negative_impls)]
#![feature(type_privacy_lints)]
#![deny(private_interfaces)]

// In this test both old and new private-in-public diagnostic were emitted.
// Old diagnostic will be deleted soon.
// See https://rust-lang.github.io/rfcs/2145-type-privacy.html.

pub trait PubPrincipal {}
auto trait PrivNonPrincipal {}

pub fn leak_dyn_nonprincipal() -> Box<dyn PubPrincipal + PrivNonPrincipal> { loop {} }
//~^ WARN private trait `PrivNonPrincipal` in public interface
//~| WARN this was previously accepted
//~| ERROR trait `PrivNonPrincipal` is more private than the item `leak_dyn_nonprincipal`

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
