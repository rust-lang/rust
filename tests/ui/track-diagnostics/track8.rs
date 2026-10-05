//@ compile-flags: -Z track-diagnostics
//@ forbid-output:interface
//@ dont-require-annotations: NOTE
#![warn(unreachable_cfg_select_predicates)]

// Normalize the emitted location so this doesn't need
// updating everytime someone adds or removes a line.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"
//@ normalize-stderr: "/rustc(?:-dev)?/[a-z0-9.]+/" -> ""

cfg_select!{
    _ => {},
    _ => {}, //~ WARN unreachable configuration predicate
}

#[repr(ferris)] //~ ERROR: malformed `repr` attribute input
#[diagnostic::ferris] //~ WARN unknown diagnostic attribute
#[doc(ferris)] //~ WARN unknown `doc` attribute `ferris`
fn main(){
    // Also check locations for buffered lints
    #[doc(ferris)] //~ WARN unknown `doc` attribute `ferris`
    println!();
}
