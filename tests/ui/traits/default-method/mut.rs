// build-pass

// pretty-expanded FIXME #23616



trait Foo {
    fn foo(&self, mut v: isize) { v = 1; }
}

pub fn main() {}
