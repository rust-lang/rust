// compile-flags: -O --target=avr-unknown-gnu-atmega328 --crate-type=rlib
// needs-llvm-components: avr

// This test validates that function pointers can be stored in global variables
// and called upon. It ensures that Rust emits function pointers in the correct
// address space to LLVM so that an assertion error relating to casting is
// not triggered.
//
// It also validates that functions can be called through function pointers
// through traits.

#![feature(no_core, lang_items, intrinsics, unboxed_closures, arbitrary_self_types)]
#![crate_type = "lib"]
#![no_core]

#[lang = "sized"]
pub trait Sized { }
#[lang = "copy"]
pub trait Copy { }
#[lang = "receiver"]
pub trait Receiver { }
#[lang = "tuple_trait"]
pub trait Tuple { }

pub struct Result<T, E> { _a: T, _b: E }

impl Copy for usize {}
impl Copy for &usize {}

#[lang = "drop_in_place"]
pub unsafe fn drop_in_place<T: ?Sized>(_: *mut T) {}

#[lang = "fn_once"]
pub trait FnOnce<Args: Tuple> {
    #[lang = "fn_once_output"]
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

#[lang = "fn_mut"]
pub trait FnMut<Args: Tuple> : FnOnce<Args> {
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

#[lang = "fn"]
pub trait Fn<Args: Tuple>: FnOnce<Args> {
    /// Performs the call operation.
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}

extern "rust-intrinsic" {
    pub fn transmute<Src, Dst>(src: Src) -> Dst;
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

// CHECK: define dso_local void @test(){{.+}}addrspace(1)
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

// Validate that we can codegen transmutes between data ptrs and fn ptrs.

// CHECK: define{{.+}}{{void \(\) addrspace\(1\)\*|ptr addrspace\(1\)}} @transmute_data_ptr_to_fn({{\{\}\*|ptr}}{{.*}} %x)
#[no_mangle]
pub unsafe fn transmute_data_ptr_to_fn(x: *const ()) -> fn() {
    // It doesn't matter precisely how this is codegenned (through memory or an addrspacecast),
    // as long as it doesn't cause a verifier error by using `bitcast`.
    transmute(x)
}

// CHECK: define{{.+}}{{\{\}\*|ptr}} @transmute_fn_ptr_to_data({{void \(\) addrspace\(1\)\*|ptr addrspace\(1\)}}{{.*}} %x)
#[no_mangle]
pub unsafe fn transmute_fn_ptr_to_data(x: fn()) -> *const () {
    // It doesn't matter precisely how this is codegenned (through memory or an addrspacecast),
    // as long as it doesn't cause a verifier error by using `bitcast`.
    transmute(x)
}

pub enum Either<T, U> { A(T), B(U) }

// Previously, we would codegen this as passing/returning a scalar pair of `{ i8, ptr }`,
// with the `ptr` field representing both `&i32` and `fn()` depending on the variant.
// This is incorrect, because `fn()` should be `ptr addrspace(1)`, not `ptr`.

// CHECK: define{{.+}}void @should_not_combine_addrspace({{.+\*|ptr}}{{.+}}sret{{.+}}%_0, {{.+\*|ptr}}{{.+}}%x)
#[no_mangle]
#[inline(never)]
pub fn should_not_combine_addrspace(x: Either<&i32, fn()>) -> Either<&i32, fn()> {
    x
}

// The incorrectness described above would result in us producing (after optimizations)
// a `ptrtoint`/`inttoptr` roundtrip to convert from `ptr` to `ptr addrspace(1)`.

// CHECK-LABEL: @call_with_fn_ptr
#[no_mangle]
pub fn call_with_fn_ptr<'a>(f: fn()) -> Either<&'a i32, fn()> {
    // CHECK-NOT: ptrtoint
    // CHECK-NOT: inttoptr
    // CHECK: call addrspace(1) void @should_not_combine_addrspace
    should_not_combine_addrspace(Either::B(f))
}
