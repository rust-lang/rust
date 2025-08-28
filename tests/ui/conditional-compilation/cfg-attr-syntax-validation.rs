#[cfg]
//~^ ERROR malformed `cfg` attribute
//~| NOTE expected this to be a list
//~| NOTE for more information, visit
struct S1;

#[cfg = 10]
//~^ ERROR malformed `cfg` attribute
//~| NOTE expected this to be a list
//~| NOTE for more information, visit
struct S2;

#[cfg()]
//~^ ERROR malformed `cfg` attribute
//~| NOTE expected a single argument here
//~| NOTE for more information, visit
struct S3;

#[cfg(a, b)]
//~^ ERROR malformed `cfg` attribute
//~| NOTE expected a single argument here
//~| NOTE for more information, visit
struct S4;

#[cfg("str")] //~ ERROR `cfg` predicate key must be an identifier
struct S5;

#[cfg(a::b)] //~ ERROR `cfg` predicate key must be an identifier
struct S6;

#[cfg(a())] //~ ERROR invalid predicate `a`
struct S7;

#[cfg(a = 10)] //~ ERROR malformed `cfg` attribute input
//~^ NOTE expected a string literal here
//~| NOTE for more information, visit
struct S8;

#[cfg(a = b"hi")]  //~ ERROR malformed `cfg` attribute input
//~^ NOTE expected a normal string literal, not a byte string literal
struct S9;

macro_rules! generate_s10 {
    ($expr: expr) => {
        #[cfg(feature = $expr)]
        //~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `expr` metavariable
        struct S10;
    }
}

generate_s10!(concat!("nonexistent"));
//~^ NOTE in this expansion of generate_s10!

fn main() {}
