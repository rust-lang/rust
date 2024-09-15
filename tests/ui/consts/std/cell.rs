use std::cell::*;

// not ok, because this creates a dangling pointer, just like `let x = Cell::new(42).as_ptr()` would
static FOO: Wrap<*mut u32> = Wrap(Cell::new(42).as_ptr());
//~^ ERROR encountered dangling pointer
const FOO_CONST: Wrap<*mut u32> = Wrap(Cell::new(42).as_ptr());
//~^ ERROR encountered dangling pointer

// Ok, these are just base values and it is the `Wrap` author's job to uphold `Send` and `Sync`
// invariants, since they used `unsafe impl`.
static FOO3: Wrap<Cell<u32>> = Wrap(Cell::new(42));
const FOO3_CONST: Wrap<Cell<u32>> = Wrap(Cell::new(42));

// ok, we are referring to the memory of another static item.
static FOO4: Wrap<*mut u32> = Wrap(FOO3.0.as_ptr());

// not ok, the use of a constant here is equivalent to an inline declaration of the value, so
// its memory will get freed before the constant is finished evaluating, thus creating a dangling
// pointer. This would happen exactly the same at runtime.
const FOO4_CONST: Wrap<*mut u32> = Wrap(FOO3_CONST.0.as_ptr());
//~^ ERROR encountered dangling pointer

// not ok, because the `as_ptr` call takes a reference to a temporary that will get freed
// before the constant is finished evaluating.
const FOO2: *mut u32 = Cell::new(42).as_ptr();
//~^ ERROR encountered dangling pointer

struct IMSafeTrustMe(UnsafeCell<u32>);
unsafe impl Send for IMSafeTrustMe {}
unsafe impl Sync for IMSafeTrustMe {}

static BAR: IMSafeTrustMe = IMSafeTrustMe(UnsafeCell::new(5));



struct Wrap<T>(T);
unsafe impl<T> Send for Wrap<T> {}
unsafe impl<T> Sync for Wrap<T> {}

static BAR_PTR: Wrap<*mut u32> = Wrap(BAR.0.get());

const fn fst_ref<T, U>(x: &(T, U)) -> &T { &x.0 }

fn main() {}
