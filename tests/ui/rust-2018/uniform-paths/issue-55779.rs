//@ check-pass
//@ edition:2018
//@ aux-crate:issue_55779_extern_trait=issue-55779-extern-trait.rs

#![allow(non_local_definitions)]

use issue_55779_extern_trait::Trait;

struct Local;
struct Helper;

impl Trait for Local {
    fn no_op(&self)
    {
        // (Unused) extern crate declaration necessary to reproduce bug
        extern crate issue_55779_extern_trait;

        // This one works
        // impl Trait for Helper { fn no_op(&self) { } }

        // This one infinite-loops
        const _IMPL_SERIALIZE_FOR_HELPER: () = {
            // (extern crate can also appear here to reproduce bug,
            // as in originating example from serde)
            impl Trait for Helper { fn no_op(&self) { } }
        };

    }
}

fn main() { }
