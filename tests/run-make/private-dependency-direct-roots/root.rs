#![crate_type = "rlib"]
#![deny(exported_private_dependencies)]

// Resolve in reverse lexical order to verify that the diagnostic sorts the roots.
extern crate right;
extern crate left;

pub fn leaks_leaf() -> left::Leaf {
    left::Leaf
}
