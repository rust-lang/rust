#![feature(associated_type_defaults)]
#![allow(clippy::needless_pass_by_value, clippy::vec_box, clippy::useless_vec)]

type Alias = Vec<Vec<Box<(u32, u32, u32, u32)>>>; // no warning here

const CST: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0))));
//~^ type_complexity

static ST: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0))));
//~^ type_complexity

struct S {
    f: Vec<Vec<Box<(u32, u32, u32, u32)>>>,
    //~^ type_complexity
}

struct Ts(Vec<Vec<Box<(u32, u32, u32, u32)>>>);
//~^ type_complexity

enum E {
    Tuple(Vec<Vec<Box<(u32, u32, u32, u32)>>>),
    //~^ type_complexity
    Struct { f: Vec<Vec<Box<(u32, u32, u32, u32)>>> },
    //~^ type_complexity
}

impl S {
    const A: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0))));
    //~^ type_complexity

    fn impl_method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>) {}
    //~^ type_complexity
}

trait T {
    const A: Vec<Vec<Box<(u32, u32, u32, u32)>>>;
    //~^ type_complexity

    type B = Vec<Vec<Box<(u32, u32, u32, u32)>>>;
    //~^ type_complexity

    fn method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>);
    //~^ type_complexity

    fn def_method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>) {}
    //~^ type_complexity
}

// Should not warn since there is likely no way to simplify this (#1013)
impl T for () {
    const A: Vec<Vec<Box<(u32, u32, u32, u32)>>> = vec![];

    type B = Vec<Vec<Box<(u32, u32, u32, u32)>>>;

    fn method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>) {}
}

fn test1() -> Vec<Vec<Box<(u32, u32, u32, u32)>>> {
    //~^ type_complexity

    vec![]
}

fn test2(_x: Vec<Vec<Box<(u32, u32, u32, u32)>>>) {}
//~^ type_complexity

fn test3() {
    let _y: Vec<Vec<Box<(u32, u32, u32, u32)>>> = vec![];
    //~^ type_complexity
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
