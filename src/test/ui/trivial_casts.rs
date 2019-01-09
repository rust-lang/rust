// Test the trivial_casts and trivial_numeric_casts lints. For each error we also
// check that the cast can be done using just coercion.

#![deny(trivial_casts, trivial_numeric_casts)]

trait Foo {
    fn foo(&self) {}
}

pub struct Bar;

impl Foo for Bar {}

pub fn main() {
    // Numeric
    let _ = 42_i32 as i32; //~ ERROR trivial numeric cast: `i32` as `i32`
    let _: i32 = 42_i32;

    let _ = 42_u8 as u8; //~ ERROR trivial numeric cast: `u8` as `u8`
    let _: u8 = 42_u8;

    // & to * pointers
    let x: &u32 = &42;
    let _ = x as *const u32; //~ERROR trivial cast: `&u32` as `*const u32`
    let _: *const u32 = x;

    let x: &mut u32 = &mut 42;
    let _ = x as *mut u32; //~ERROR trivial cast: `&mut u32` as `*mut u32`
    let _: *mut u32 = x;

    // unsize array
    let x: &[u32; 3] = &[42, 43, 44];
    let _ = x as &[u32]; //~ERROR trivial cast: `&[u32; 3]` as `&[u32]`
    let _ = x as *const [u32]; //~ERROR trivial cast: `&[u32; 3]` as `*const [u32]`
    let _: &[u32] = x;
    let _: *const [u32] = x;

    let x: &mut [u32; 3] = &mut [42, 43, 44];
    let _ = x as &mut [u32]; //~ERROR trivial cast: `&mut [u32; 3]` as `&mut [u32]`
    let _ = x as *mut [u32]; //~ERROR trivial cast: `&mut [u32; 3]` as `*mut [u32]`
    let _: &mut [u32] = x;
    let _: *mut [u32] = x;

    let x: Box<[u32; 3]> = Box::new([42, 43, 44]);
    let _ = x as Box<[u32]>;
    //~^ ERROR trivial cast: `std::boxed::Box<[u32; 3]>` as `std::boxed::Box<[u32]>`
    let x: Box<[u32; 3]> = Box::new([42, 43, 44]);
    let _: Box<[u32]> = x;

    // unsize trait
    let x: &Bar = &Bar;
    let _ = x as &Foo; //~ERROR trivial cast: `&Bar` as `&dyn Foo`
    let _ = x as *const Foo; //~ERROR trivial cast: `&Bar` as `*const dyn Foo`
    let _: &Foo = x;
    let _: *const Foo = x;

    let x: &mut Bar = &mut Bar;
    let _ = x as &mut Foo; //~ERROR trivial cast: `&mut Bar` as `&mut dyn Foo`
    let _ = x as *mut Foo; //~ERROR trivial cast: `&mut Bar` as `*mut dyn Foo`
    let _: &mut Foo = x;
    let _: *mut Foo = x;

    let x: Box<Bar> = Box::new(Bar);
    let _ = x as Box<Foo>; //~ERROR `std::boxed::Box<Bar>` as `std::boxed::Box<dyn Foo>`
    let x: Box<Bar> = Box::new(Bar);
    let _: Box<Foo> = x;

    // functions
    fn baz(_x: i32) {}
    let _ = &baz as &Fn(i32); //~ERROR `&fn(i32) {main::baz}` as `&dyn std::ops::Fn(i32)`
    let _: &Fn(i32) = &baz;
    let x = |_x: i32| {};
    let _ = &x as &Fn(i32); //~ERROR trivial cast
    let _: &Fn(i32) = &x;
}

// subtyping
pub fn test_subtyping<'a, 'b: 'a>(a: &'a Bar, b: &'b Bar) {
    let _ = a as &'a Bar; //~ERROR trivial cast
    let _: &'a Bar = a;
    let _ = b as &'a Bar; //~ERROR trivial cast
    let _: &'a Bar = b;
    let _ = b as &'b Bar; //~ERROR trivial cast
    let _: &'b Bar = b;
}
