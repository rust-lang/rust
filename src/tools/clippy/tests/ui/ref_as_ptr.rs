#![warn(clippy::ref_as_ptr)]
#![allow(clippy::unnecessary_mut_passed, clippy::needless_lifetimes)]

fn f<T>(_: T) {}

fn main() {
    f(&1u8 as *const _);
    //~^ ref_as_ptr
    f(&2u32 as *const u32);
    //~^ ref_as_ptr
    f(&3.0f64 as *const f64);
    //~^ ref_as_ptr

    f(&4 as *const _ as *const f32);
    //~^ ref_as_ptr
    f(&5.0f32 as *const f32 as *const u32);
    //~^ ref_as_ptr

    f(&mut 6u8 as *const _);
    //~^ ref_as_ptr
    f(&mut 7u32 as *const u32);
    //~^ ref_as_ptr
    f(&mut 8.0f64 as *const f64);
    //~^ ref_as_ptr

    f(&mut 9 as *const _ as *const f32);
    //~^ ref_as_ptr
    f(&mut 10.0f32 as *const f32 as *const u32);
    //~^ ref_as_ptr

    f(&mut 11u8 as *mut _);
    //~^ ref_as_ptr
    f(&mut 12u32 as *mut u32);
    //~^ ref_as_ptr
    f(&mut 13.0f64 as *mut f64);
    //~^ ref_as_ptr

    f(&mut 14 as *mut _ as *const f32);
    //~^ ref_as_ptr
    f(&mut 15.0f32 as *mut f32 as *const u32);
    //~^ ref_as_ptr

    f(&1u8 as *const _);
    //~^ ref_as_ptr
    f(&2u32 as *const u32);
    //~^ ref_as_ptr
    f(&3.0f64 as *const f64);
    //~^ ref_as_ptr

    f(&4 as *const _ as *const f32);
    //~^ ref_as_ptr
    f(&5.0f32 as *const f32 as *const u32);
    //~^ ref_as_ptr

    let val = 1;
    f(&val as *const _);
    //~^ ref_as_ptr
    f(&val as *const i32);
    //~^ ref_as_ptr

    f(&val as *const _ as *const f32);
    //~^ ref_as_ptr
    f(&val as *const i32 as *const f64);
    //~^ ref_as_ptr

    let mut val: u8 = 2;
    f(&mut val as *mut u8);
    //~^ ref_as_ptr
    f(&mut val as *mut _);
    //~^ ref_as_ptr

    f(&mut val as *const u8);
    //~^ ref_as_ptr
    f(&mut val as *const _);
    //~^ ref_as_ptr

    f(&mut val as *const u8 as *const f64);
    //~^ ref_as_ptr
    f::<*const Option<u8>>(&mut val as *const _ as *const _);
    //~^ ref_as_ptr

    f(&std::array::from_fn(|i| i * i) as *const [usize; 7]);
    //~^ ref_as_ptr
    f(&mut std::array::from_fn(|i| i * i) as *const [usize; 8]);
    //~^ ref_as_ptr
    f(&mut std::array::from_fn(|i| i * i) as *mut [usize; 9]);
    //~^ ref_as_ptr

    let _ = &String::new() as *const _;
    let _ = &mut String::new() as *mut _;
    const FOO: *const String = &String::new() as *const _;
}

#[clippy::msrv = "1.75"]
fn _msrv_1_75() {
    let val = &42_i32;
    let mut_val = &mut 42_i32;

    // `std::ptr::from_{ref, mut}` was stabilized in 1.76. Do not lint this
    f(val as *const i32);
    f(mut_val as *mut i32);
}

#[clippy::msrv = "1.76"]
fn _msrv_1_76() {
    let val = &42_i32;
    let mut_val = &mut 42_i32;

    f(val as *const i32);
    //~^ ref_as_ptr
    f(mut_val as *mut i32);
    //~^ ref_as_ptr
}

fn foo(val: &[u8]) {
    f(val as *const _);
    //~^ ref_as_ptr
    f(val as *const [u8]);
    //~^ ref_as_ptr
}

fn bar(val: &mut str) {
    f(val as *mut _);
    //~^ ref_as_ptr
    f(val as *mut str);
    //~^ ref_as_ptr
}

struct X<'a>(&'a i32);

impl<'a> X<'a> {
    fn foo(&self) -> *const i64 {
        self.0 as *const _ as *const _
        //~^ ref_as_ptr
    }

    fn bar(&mut self) -> *const i64 {
        self.0 as *const _ as *const _
        //~^ ref_as_ptr
    }
}

struct Y<'a>(&'a mut i32);

impl<'a> Y<'a> {
    fn foo(&self) -> *const i64 {
        self.0 as *const _ as *const _
        //~^ ref_as_ptr
    }

    fn bar(&mut self) -> *const i64 {
        self.0 as *const _ as *const _
        //~^ ref_as_ptr
    }

    fn baz(&mut self) -> *const i64 {
        self.0 as *mut _ as *mut _
        //~^ ref_as_ptr
    }
}
