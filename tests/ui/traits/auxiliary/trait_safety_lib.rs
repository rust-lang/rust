// Simple smoke test that unsafe traits can be compiled etc.

pub unsafe trait Foo {
    fn foo(&self) -> isize;
}

unsafe impl Foo for isize {
    fn foo(&self) -> isize { *self }
}
