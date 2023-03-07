// run-pass
// Simple smoke test that unsafe traits can be compiled etc.


unsafe trait Foo {
    fn foo(&self) -> isize;
}

unsafe impl Foo for isize {
    fn foo(&self) -> isize { *self }
}

fn take_foo<F:Foo>(f: &F) -> isize { f.foo() }

fn main() {
    let x: isize = 22;
    assert_eq!(22, take_foo(&x));
}
