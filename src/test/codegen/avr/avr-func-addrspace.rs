// compile-flags: -O --target=avr-unknown-gnu-atmega328 --crate-type=rlib
// needs-llvm-components: avr

// This test validates that function pointers can be stored in global variables
// and called upon. It ensures that Rust emits function pointers in the correct
// address space to LLVM so that an assertion error relating to casting is
// not triggered.
//
// It also validates that functions can be called through function pointers
// through traits.

#![feature(no_core, lang_items, unboxed_closures, arbitrary_self_types)]
#![crate_type = "lib"]
#![no_core]

#[lang = "sized"]
pub trait Sized { }
#[lang = "copy"]
pub trait Copy { }
#[lang = "receiver"]
pub trait Receiver { }

pub struct Result<T, E> { _a: T, _b: E }

impl Copy for usize {}
impl Copy for &usize {}

#[lang = "drop_in_place"]
pub unsafe fn drop_in_place<T: ?Sized>(_: *mut T) {}

#[lang = "fn_once"]
pub trait FnOnce<Args> {
    #[lang = "fn_once_output"]
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

#[lang = "fn_mut"]
pub trait FnMut<Args> : FnOnce<Args> {
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

#[lang = "fn"]
pub trait Fn<Args>: FnOnce<Args> {
    /// Performs the call operation.
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}

impl<'a, A, R> FnOnce<A> for &'a fn(A) -> R {
    type Output = R;

    extern "rust-call" fn call_once(self, args: A) -> R {
        (*self)(args)
    }
}

pub static mut STORAGE_FOO: fn(&usize, &mut u32) -> Result<(), ()> = arbitrary_black_box;
pub static mut STORAGE_BAR: u32 = 12;

fn arbitrary_black_box(ptr: &usize, _: &mut u32) -> Result<(), ()> {
    let raw_ptr = ptr as *const usize;
    let _v: usize = unsafe { *raw_ptr };
    loop {}
}

#[inline(never)]
#[no_mangle]
fn call_through_fn_trait(a: &mut impl Fn<(), Output=()>) {
    (*a)()
}

#[inline(never)]
fn update_bar_value() {
    unsafe {
        STORAGE_BAR = 88;
    }
}

// CHECK: define void @test(){{.+}}addrspace(1)
#[no_mangle]
pub extern "C" fn test() {
    let mut buf = 7;

    // A call through the Fn trait must use address space 1.
    //
    // CHECK: call{{.+}}addrspace(1) void @call_through_fn_trait()
    call_through_fn_trait(&mut update_bar_value);

    // A call through a global variable must use address space 1.
    // CHECK: load {{.*}}addrspace(1){{.+}}FOO
    unsafe {
        STORAGE_FOO(&1, &mut buf);
    }
}
