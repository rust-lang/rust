// run-pass

// This test unsures that with_opt_const_param returns the
// def_id of the N param in the Foo::Assoc GAT.

trait Foo {
    type Assoc<const N: usize>;
    fn foo(&self) -> Self::Assoc<3>;
}

impl Foo for () {
    type Assoc<const N: usize> = [(); N];
    fn foo(&self) -> Self::Assoc<3> {
        [(); 3]
    }
}

fn main() {
    assert_eq!(().foo(), [(); 3]);
}
