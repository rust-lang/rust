//@compile-flags: -Zdeduplicate-diagnostics=yes

#![warn(clippy::all)]
#![allow(unused, clippy::needless_pass_by_value, clippy::vec_box, clippy::useless_vec)]
#![feature(associated_type_defaults)]

type Alias = Vec<Vec<Box<(u32, u32, u32, u32)>>>; // no warning here

const CST: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0))));
//~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
//~| NOTE: `-D clippy::type-complexity` implied by `-D warnings`
static ST: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0))));
//~^ ERROR: very complex type used. Consider factoring parts into `type` definitions

struct S {
    f: Vec<Vec<Box<(u32, u32, u32, u32)>>>,
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
}

struct Ts(Vec<Vec<Box<(u32, u32, u32, u32)>>>);
//~^ ERROR: very complex type used. Consider factoring parts into `type` definitions

enum E {
    Tuple(Vec<Vec<Box<(u32, u32, u32, u32)>>>),
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
    Struct { f: Vec<Vec<Box<(u32, u32, u32, u32)>>> },
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
}

impl S {
    const A: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0))));
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
    fn impl_method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>) {}
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
}

trait T {
    const A: Vec<Vec<Box<(u32, u32, u32, u32)>>>;
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
    type B = Vec<Vec<Box<(u32, u32, u32, u32)>>>;
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
    fn method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>);
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
    fn def_method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>) {}
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
}

// Should not warn since there is likely no way to simplify this (#1013)
impl T for () {
    const A: Vec<Vec<Box<(u32, u32, u32, u32)>>> = vec![];

    type B = Vec<Vec<Box<(u32, u32, u32, u32)>>>;

    fn method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>) {}
}

fn test1() -> Vec<Vec<Box<(u32, u32, u32, u32)>>> {
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
    vec![]
}

fn test2(_x: Vec<Vec<Box<(u32, u32, u32, u32)>>>) {}
//~^ ERROR: very complex type used. Consider factoring parts into `type` definitions

fn test3() {
    let _y: Vec<Vec<Box<(u32, u32, u32, u32)>>> = vec![];
    //~^ ERROR: very complex type used. Consider factoring parts into `type` definitions
}

#[repr(C)]
struct D {
    // should not warn, since we don't have control over the signature (#3222)
    test4: extern "C" fn(
        itself: &D,
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        e: usize,
        f: usize,
        g: usize,
        h: usize,
        i: usize,
    ),
}

fn main() {}
