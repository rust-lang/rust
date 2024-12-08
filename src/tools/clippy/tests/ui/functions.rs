#![warn(clippy::all)]
#![allow(dead_code, unused_unsafe)]
#![allow(clippy::missing_safety_doc, clippy::uninlined_format_args)]

// TOO_MANY_ARGUMENTS
fn good(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool) {}

fn bad(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool, _eight: ()) {}
//~^ ERROR: this function has too many arguments (8/7)
//~| NOTE: `-D clippy::too-many-arguments` implied by `-D warnings`

#[rustfmt::skip]
fn bad_multiline(
//~^ ERROR: this function has too many arguments (8/7)
    one: u32,
    two: u32,
    three: &str,
    four: bool,
    five: f32,
    six: f32,
    seven: bool,
    eight: ()
) {
    let _one = one;
    let _two = two;
    let _three = three;
    let _four = four;
    let _five = five;
    let _six = six;
    let _seven = seven;
}

// don't lint extern fns
extern "C" fn extern_fn(
    _one: u32,
    _two: u32,
    _three: *const u8,
    _four: bool,
    _five: f32,
    _six: f32,
    _seven: bool,
    _eight: *const std::ffi::c_void,
) {
}

pub trait Foo {
    fn good(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool);
    fn bad(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool, _eight: ());
    //~^ ERROR: this function has too many arguments (8/7)

    fn ptr(p: *const u8);
}

pub struct Bar;

impl Bar {
    fn good_method(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool) {}
    fn bad_method(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool, _eight: ()) {}
    //~^ ERROR: this function has too many arguments (8/7)
}

// ok, we don’t want to warn implementations
impl Foo for Bar {
    fn good(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool) {}
    fn bad(_one: u32, _two: u32, _three: &str, _four: bool, _five: f32, _six: f32, _seven: bool, _eight: ()) {}

    fn ptr(p: *const u8) {
        println!("{}", unsafe { *p });
        //~^ ERROR: this public function might dereference a raw pointer but is not marked
        //~| NOTE: `-D clippy::not-unsafe-ptr-arg-deref` implied by `-D warnings`
        println!("{:?}", unsafe { p.as_ref() });
        //~^ ERROR: this public function might dereference a raw pointer but is not marked
        unsafe { std::ptr::read(p) };
        //~^ ERROR: this public function might dereference a raw pointer but is not marked
    }
}

// NOT_UNSAFE_PTR_ARG_DEREF

fn private(p: *const u8) {
    println!("{}", unsafe { *p });
}

pub fn public(p: *const u8) {
    println!("{}", unsafe { *p });
    //~^ ERROR: this public function might dereference a raw pointer but is not marked `un
    println!("{:?}", unsafe { p.as_ref() });
    //~^ ERROR: this public function might dereference a raw pointer but is not marked `un
    unsafe { std::ptr::read(p) };
    //~^ ERROR: this public function might dereference a raw pointer but is not marked `un
}

type Alias = *const u8;

pub fn type_alias(p: Alias) {
    println!("{}", unsafe { *p });
    //~^ ERROR: this public function might dereference a raw pointer but is not marked `un
    println!("{:?}", unsafe { p.as_ref() });
    //~^ ERROR: this public function might dereference a raw pointer but is not marked `un
    unsafe { std::ptr::read(p) };
    //~^ ERROR: this public function might dereference a raw pointer but is not marked `un
}

impl Bar {
    fn private(self, p: *const u8) {
        println!("{}", unsafe { *p });
    }

    pub fn public(self, p: *const u8) {
        println!("{}", unsafe { *p });
        //~^ ERROR: this public function might dereference a raw pointer but is not marked
        println!("{:?}", unsafe { p.as_ref() });
        //~^ ERROR: this public function might dereference a raw pointer but is not marked
        unsafe { std::ptr::read(p) };
        //~^ ERROR: this public function might dereference a raw pointer but is not marked
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
