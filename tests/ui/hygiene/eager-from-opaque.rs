// Opaque macro can eagerly expand its input without breaking its resolution.
// Regression test for issue #63685.

//@ check-pass

macro_rules! foo {
    () => {
        "foo"
    };
}

macro_rules! bar {
    () => {
        foo!()
    };
}

fn main() {
    format_args!(bar!());
}
