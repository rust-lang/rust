// edition:2018

// In this test baz isn't resolved when called as foo.baz even though
// it's called from inside foo. This is somewhat surprising and may
// want to change eventually.

mod foo {
    pub fn bar() { foo::baz(); } //~ ERROR failed to resolve: use of undeclared crate or module `foo`

    fn baz() { }
}

fn main() { }
