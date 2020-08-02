// aux-build:custom-quote.rs
// aux-build:span-from-proc-macro.rs
// compile-flags: -Z macro-backtrace

#[macro_use]
extern crate span_from_proc_macro;

#[error_from_attribute] //~ ERROR cannot find type `MissingType`
struct ShouldBeRemoved;

#[derive(ErrorFromDerive)] //~ ERROR cannot find type `OtherMissingType`
struct Kept;

fn main() {
    error_from_bang!(); //~ ERROR mismatched types
    other_error_from_bang!(); //~ ERROR cannot find value `my_ident`
}
