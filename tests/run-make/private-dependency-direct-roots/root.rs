#![crate_type = "rlib"]
#![deny(exported_private_dependencies)]

// Load both private roots, but select only `left` in the public interface.
extern crate left;
extern crate right;

pub fn leaks_leaf() -> left::Leaf {
    left::Leaf
}
