// Test that we correctly infer variance for region parameters in
// case that involve multiple intricate types.
// Try enums too.

#![feature(rustc_attrs)]

#[rustc_variance]
enum Base<'a, 'b, 'c:'b, 'd> { //~ ERROR ['a: -, 'b: +, 'c: o, 'd: *]
    //~^ ERROR: `'d` is never used
    Test8A(extern "Rust" fn(&'a isize)),
    Test8B(&'b [isize]),
    Test8C(&'b mut &'c str),
}

#[rustc_variance]
struct Derived1<'w, 'x:'y, 'y, 'z> { //~ ERROR ['w: *, 'x: o, 'y: +, 'z: -]
    //~^ ERROR: `'w` is never used
    f: Base<'z, 'y, 'x, 'w>
}

#[rustc_variance] // Combine - and + to yield o
struct Derived2<'a, 'b:'a, 'c> { //~ ERROR ['a: o, 'b: o, 'c: *]
    //~^ ERROR: `'c` is never used
    f: Base<'a, 'a, 'b, 'c>
}

#[rustc_variance] // Combine + and o to yield o (just pay attention to 'a here)
struct Derived3<'a:'b, 'b, 'c> { //~ ERROR ['a: o, 'b: +, 'c: *]
    //~^ ERROR: `'c` is never used
    f: Base<'a, 'b, 'a, 'c>
}

#[rustc_variance] // Combine + and * to yield + (just pay attention to 'a here)
struct Derived4<'a, 'b, 'c:'b> { //~ ERROR ['a: -, 'b: +, 'c: o]
    f: Base<'a, 'b, 'c, 'a>
}

fn main() {}
