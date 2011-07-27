// error-pattern: unresolved name

// In this test baz isn't resolved when called as foo.baz even though
// it's called from inside foo. This is somewhat surprising and may
// want to change eventually.

mod foo {

    export bar;

    fn bar() { foo::baz(); }

    fn baz() { }
}

fn main() { }