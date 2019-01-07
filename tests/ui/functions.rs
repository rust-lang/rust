#![warn(clippy::all)]
#![allow(dead_code)]
#![allow(unused_unsafe)]

// TOO_MANY_ARGUMENTS
fn good(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool) {}

fn bad(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool, _eight: ()) {}

// don't lint extern fns
extern "C" fn extern_fn(
    _one: u32,
    _two: u32,
    _three: &str,
    _four: bool,
    _five: f32,
    _six: f32,
    _seven: bool,
    _eight: (),
) {
}

pub trait Foo {
    fn good(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool);
    fn bad(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool, _eight: ());

    fn ptr(p: *const u8);
}

pub struct Bar;

impl Bar {
    fn good_method(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool) {}
    fn bad_method(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool, _eight: ()) {}
}

// ok, we don’t want to warn implementations
impl Foo for Bar {
    fn good(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool) {}
    fn bad(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool, _eight: ()) {}

    fn ptr(p: *const u8) {
        println!("{}", unsafe { *p });
        println!("{:?}", unsafe { p.as_ref() });
        unsafe { std::ptr::read(p) };
    }
}

// NOT_UNSAFE_PTR_ARG_DEREF

fn private(p: *const u8) {
    println!("{}", unsafe { *p });
}

pub fn public(p: *const u8) {
    println!("{}", unsafe { *p });
    println!("{:?}", unsafe { p.as_ref() });
    unsafe { std::ptr::read(p) };
}

impl Bar {
    fn private(self, p: *const u8) {
        println!("{}", unsafe { *p });
    }

    pub fn public(self, p: *const u8) {
        println!("{}", unsafe { *p });
        println!("{:?}", unsafe { p.as_ref() });
        unsafe { std::ptr::read(p) };
    }

    pub fn public_ok(self, p: *const u8) {
        if !p.is_null() {
            println!("{:p}", p);
        }
    }

    pub unsafe fn public_unsafe(self, p: *const u8) {
        println!("{}", unsafe { *p });
        println!("{:?}", unsafe { p.as_ref() });
    }
}

fn main() {}
