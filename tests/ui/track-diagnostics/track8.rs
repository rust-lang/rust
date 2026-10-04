//@ compile-flags: -Z track-diagnostics
//@ forbid-output:interface
//@ dont-require-annotations: NOTE
#![warn(unreachable_cfg_select_predicates)]

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
