// issue#140255

#[macro_use::a]       //~ ERROR failed to resolve: use of unresolved module or unlinked crate `macro_use`
fn f0() {}

#[macro_use::a::b]    //~ ERROR failed to resolve: use of unresolved module or unlinked crate `macro_use`
fn f1() {}

#[macro_escape::a]    //~ ERROR failed to resolve: use of unresolved module or unlinked crate `macro_escape`
fn f2() {}

#[macro_escape::a::b] //~ ERROR failed to resolve: use of unresolved module or unlinked crate `macro_escape`
fn f3() {}

fn main() {}
