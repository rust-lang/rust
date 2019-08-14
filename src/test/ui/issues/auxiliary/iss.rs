#![crate_name="issue6919_3"]

// part of issue-6919.rs

pub struct C<K> where K: FnOnce() {
    pub k: K,
}

fn no_op() { }
pub const D : C<fn()> = C {
    k: no_op as fn()
};
