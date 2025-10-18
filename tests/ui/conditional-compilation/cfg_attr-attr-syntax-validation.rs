#[cfg_attr]
//~^ ERROR malformed `cfg_attr` attribute
struct S1;

#[cfg_attr = 10]
//~^ ERROR malformed `cfg_attr` attribute
struct S2;

#[cfg_attr()]
//~^ ERROR malformed `cfg_attr` attribute
struct S3;

#[cfg_attr("str")] //~ ERROR malformed `cfg_attr` attribute input
struct S5;

#[cfg_attr(a::b)] //~ ERROR malformed `cfg_attr` attribute input
struct S6;

#[cfg_attr(a())] //~ ERROR invalid predicate `a`
struct S7;

#[cfg_attr(a = 10)] //~ ERROR malformed `cfg_attr` attribute input
struct S8;

#[cfg_attr(a = b"hi")]  //~ ERROR malformed `cfg_attr` attribute input
struct S9;

macro_rules! generate_s10 {
    ($expr: expr) => {
        #[cfg_attr(feature = $expr)]
        //~^ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `expr` metavariable
        struct S10;
    }
}

generate_s10!(concat!("nonexistent"));

#[cfg_attr(true)] //~ ERROR expected `,`, found end of `cfg_attr` input
struct S11;

#[cfg_attr(true, unknown_attribute)] //~ ERROR cannot find attribute `unknown_attribute` in this scope
struct S12;

#[cfg_attr(true, link_section)] //~ ERROR malformed `link_section` attribute input
//~^ WARN attribute cannot be used on
//~| WARN previously accepted
struct S13;

#[cfg_attr(true, inline())] //~ ERROR malformed `inline` attribute input
fn f1() {}

fn main() {}
