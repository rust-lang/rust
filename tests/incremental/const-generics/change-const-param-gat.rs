//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Zincremental-ignore-spans
#![feature(generic_associated_types)]

// This test unsures that with_opt_const_param returns the
// def_id of the N param in the Foo::Assoc GAT.

trait Foo {
    type Assoc<const N: usize>;
    fn foo(
        &self,
    ) -> Self::Assoc<{ if cfg!(rpass2) { 3 } else { 2 } }>;
}

impl Foo for () {
    type Assoc<const N: usize> = [(); N];
    fn foo(
        &self,
    ) -> Self::Assoc<{ if cfg!(rpass2) { 3 } else { 2 } }> {
        [(); { if cfg!(rpass2) { 3 } else { 2 } }]
    }
}

fn main() {
    assert_eq!(
        ().foo(),
        [(); { if cfg!(rpass2) { 3 } else { 2 } }]
    );
}
