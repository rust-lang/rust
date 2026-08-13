// Macro that expans to statement called in expression context of another macro argument must not
// suggest using a `;`. Issue #30597.

macro_rules! foo {
    ($s:stmt) => { $s } //~ ERROR expected expression, found `stmt` metavariable
}

fn main() {
    println!("{}", foo!(42));
    //~^ HELP surround the macro invocation with `{}` to interpret the expansion as a statement
}
