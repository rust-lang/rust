#![feature(pattern_types, rustc_attrs)]
#![feature(pattern_type_macro)]
#![feature(pattern_type_range_trait,const_trait_impl)]
#![deny(improper_c_fn_definitions)]
#![allow(unused)]
use std::pat::pattern_type;
use std::mem::transmute;

macro_rules! mini_tr {
    ($name:ident : $type:ty) => {
        let $name: $type = unsafe {transmute($name)};
    };
}

const USZM1: usize = usize::MAX -1;
const ISZP1: isize = -isize::MAX;

extern "C" fn test_me(
    // "standard" tests (see if option works as intended)
    a: pattern_type!(u32 is 0..),
    ao: Option<pattern_type!(u32 is 0..)>, //~ ERROR: not FFI-safe
    b: pattern_type!(u32 is 1..),  //~ ERROR: not FFI-safe
    bo: Option<pattern_type!(u32 is 1..)>,
    c: pattern_type!(u32 is 2..),  //~ ERROR: not FFI-safe
    co: Option<pattern_type!(u32 is 2..)>,  //~ ERROR: not FFI-safe

    // fuzz-testing (see if the disallowed-value-count function works)
    e1: Option<pattern_type!(i32 is ..0x7fffffff)>,
    e2: Option<pattern_type!(u32 is ..0xffffffff)>,
    e3: Option<pattern_type!(i128 is ..0x7fffffff_ffffffff_ffffffff_ffffffff)>,
    //~^ ERROR: uses type `i128`
    e4: Option<pattern_type!(u128 is ..0xffffffff_ffffffff_ffffffff_ffffffff)>,
    //~^ ERROR: uses type `u128`
    f1: Option<pattern_type!(i32 is -0x7fffffff..)>,
    f2: Option<pattern_type!(u32 is 1..)>,
    f3: Option<pattern_type!(i128 is -0x7fffffff_ffffffff_ffffffff_ffffffff..)>,
    //~^ ERROR: uses type `i128`
    f4: Option<pattern_type!(u128 is 1..)>,
    //~^ ERROR: uses type `u128`
    g11: Option<pattern_type!(i32 is ..-2 | -1..)>,
    g12: Option<pattern_type!(i32 is ..-1 | 0..)>,
    g13: Option<pattern_type!(i32 is ..0 | 1..)>,
    g14: Option<pattern_type!(i32 is ..1 | 2..)>,
    //g2: Option<pattern_type!(u32 is ..5 | 6..)>,
    // ^ error: only signed integer base types are allowed for or-pattern pattern types at present
    g31: Option<pattern_type!(i128 is ..-2 | -1..)>,
    //~^ ERROR: uses type `i128`
    g32: Option<pattern_type!(i128 is ..-1 | 0..)>,
    //~^ ERROR: uses type `i128`
    g33: Option<pattern_type!(i128 is ..0 | 1..)>,
    //~^ ERROR: uses type `i128`
    g34: Option<pattern_type!(i128 is ..1 | 2..)>,
    //~^ ERROR: uses type `i128`
    //g4: Option<pattern_type!(u128 is ..5 | 6..)>,
    // ^ ERROR: uses type `u128`

    // because usize patterns have "unevaluated const" implicit bounds and this needs to not ICE
    h1: pattern_type!(usize is 1..),  //~ ERROR: not FFI-safe
    h2: pattern_type!(usize is ..USZM1),  //~ ERROR: not FFI-safe
    // h3: pattern_type!(usize is ..), // not allowed
    h4: pattern_type!(isize is ISZP1..), //~ ERROR: not FFI-safe

    h: pattern_type!(char is '\0'..),
    //~^ ERROR: uses type `char`
    //~| ERROR: uses type `(char) is '\0'..`
){
    // triple-check that the options with supposed layout optimisations actually have them
    mini_tr!(bo: u32);
    mini_tr!(co: u32);

    mini_tr!(e1: i32);
    mini_tr!(e2: u32);
    mini_tr!(e3: i128);
    mini_tr!(e4: u128);
    mini_tr!(f1: i32);
    mini_tr!(f2: u32);
    mini_tr!(f3: i128);
    mini_tr!(f4: u128);

    mini_tr!(g11: i32);
    mini_tr!(g12: i32);
    mini_tr!(g13: i32);
    mini_tr!(g14: i32);
    //mini_tr!(g2: u32);
    mini_tr!(g31: i128);
    mini_tr!(g32: i128);
    mini_tr!(g33: i128);
    mini_tr!(g34: i128);
    //mini_tr!(g4: u128);
}

fn main(){}
