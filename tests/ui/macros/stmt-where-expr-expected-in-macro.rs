// Macro that expans to statement called in expression context of another macro argument must not
// suggest using a `;`. Issue #30597.

macro_rules! foo {
    ($s:stmt) => { $s } //~ ERROR expected expression, found `stmt` metavariable
}

macro_rules! bar { ($s:stmt) => { $s } } //~ ERROR expected expression, found `stmt` metavariable

fn main() {
    let _ = foo!(42);
    //~^ HELP invoke the macro in a new block with a single statement to interpret the expansion as a statement
    bar!(32)
    //~^ HELP invoke the macro in a new block with a single statement to interpret the expansion as a statement
}
