#[cfg] //~ ERROR `cfg` is not followed by parentheses
struct S1;

#[cfg = 10] //~ ERROR `cfg` is not followed by parentheses
struct S2;

#[cfg()] //~ ERROR `cfg` predicate is not specified
struct S3;

#[cfg(a, b)] //~ ERROR multiple `cfg` predicates are specified
struct S4;

#[cfg("str")] //~ ERROR `cfg` predicate key cannot be a literal
struct S5;

#[cfg(a::b)] //~ ERROR `cfg` predicate key must be an identifier
struct S6;

#[cfg(a())] //~ ERROR invalid predicate `a`
struct S7;

#[cfg(a = 10)] //~ ERROR literal in `cfg` predicate value must be a string
struct S8;

#[cfg(a = b"hi")] //~ ERROR literal in `cfg` predicate value must be a string
struct S9;

#[cfg = a] //~ ERROR attribute value must be a literal
struct S10;

#[cfg(a)] //~ WARN unexpected `cfg` condition name: `a`
struct S11;

macro_rules! generate_s10 {
    ($expr: expr) => {
        #[cfg(feature = $expr)]
        //~^ ERROR expected unsuffixed literal, found `expr` metavariable
        struct S12;
    }
}

generate_s10!(concat!("nonexistent"));

mod m {
    #![cfg] //~ ERROR `cfg` is not followed by parentheses
    #![cfg = 10] //~ ERROR `cfg` is not followed by parentheses
    #![cfg()] //~ ERROR `cfg` predicate is not specified
}
fn main() {}
