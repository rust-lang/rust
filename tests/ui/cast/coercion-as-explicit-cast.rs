//! This test checks that various forms of "trivial" casts and coercions
//! can be explicitly performed using the `as` keyword without compilation errors.

//@ run-pass

#![allow(trivial_casts, trivial_numeric_casts)]

trait Foo {
    fn foo(&self) {} //~ WARN method `foo` is never used
}

pub struct Bar;

impl Foo for Bar {}

pub fn main() {
    // Numeric
    let _ = 42_i32 as i32;
    let _ = 42_u8 as u8;

    // & to * pointers
    let x: &u32 = &42;
    let _ = x as *const u32;

    let x: &mut u32 = &mut 42;
    let _ = x as *mut u32;

    // unsize array
    let x: &[u32; 3] = &[42, 43, 44];
    let _ = x as &[u32];
    let _ = x as *const [u32];

    let x: &mut [u32; 3] = &mut [42, 43, 44];
    let _ = x as &mut [u32];
    let _ = x as *mut [u32];

    let x: Box<[u32; 3]> = Box::new([42, 43, 44]);
    let _ = x as Box<[u32]>;

    // unsize trait
    let x: &Bar = &Bar;
    let _ = x as &dyn Foo;
    let _ = x as *const dyn Foo;

    let x: &mut Bar = &mut Bar;
    let _ = x as &mut dyn Foo;
    let _ = x as *mut dyn Foo;

    let x: Box<Bar> = Box::new(Bar);
    let _ = x as Box<dyn Foo>;

    // functions
    fn baz(_x: i32) {}
    let _ = &baz as &dyn Fn(i32);
    let x = |_x: i32| {};
    let _ = &x as &dyn Fn(i32);
}

// subtyping
pub fn test_subtyping<'a, 'b: 'a>(a: &'a Bar, b: &'b Bar) {
    let _ = a as &'a Bar;
    let _ = b as &'a Bar;
    let _ = b as &'b Bar;
}
