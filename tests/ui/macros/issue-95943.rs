// Regression test for <https://github.com/rust-lang/rust/issues/95943>.
//
// When a macro RHS references a metavariable that wasn't declared in the
// matcher, the diagnostic should clearly identify the missing macro parameter
// instead of emitting the misleading "expected expression, found `$`".

macro_rules! m {
    () => {
        $x;
        //~^ ERROR cannot find macro parameter `$x` in this scope
    };
}

fn main() {
    m!()
}
