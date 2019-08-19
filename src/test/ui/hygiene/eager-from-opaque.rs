// Opaque macro can eagerly expand its input without breaking its resolution.
// Regression test for issue #63685.

macro_rules! foo {
    () => {
        "foo"
    };
}

macro_rules! bar {
    () => {
        foo!() //~ ERROR cannot find macro `foo!` in this scope
    };
}

fn main() {
    format_args!(bar!());
}
