//@ compile-flags: -Zunleash-the-miri-inside-of-you
//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ dont-require-annotations: NOTE

#![allow(static_mut_refs)]
use std::cell::UnsafeCell;
use std::sync::atomic::*;

// # Plain `&mut` in the final value

// This requires walking nested statics.
static FOO: &&mut u32 = &&mut 42;
//~^ ERROR pointing to read-only memory
static OH_YES: &mut i32 = &mut 42;
//~^ ERROR pointing to read-only memory
static BAR: &mut () = &mut ();
//~^ ERROR encountered mutable pointer in final value of static

struct Foo<T>(T);

static BOO: &mut Foo<()> = &mut Foo(());
//~^ ERROR encountered mutable pointer in final value of static

const BLUNT: &mut i32 = &mut 42;
//~^ ERROR: pointing to read-only memory

const SUBTLE: &mut i32 = unsafe {
    //~^ ERROR: encountered mutable reference
    static mut STATIC: i32 = 0;
    &mut STATIC
};

// # Interior mutability

struct Meh {
    x: &'static UnsafeCell<i32>,
}
unsafe impl Sync for Meh {}
static MEH: Meh = Meh { x: &UnsafeCell::new(42) };
//~^ ERROR `UnsafeCell` in read-only memory
// Same with a const:
// the following will never be ok! no interior mut behind consts, because
// all allocs interned here will be marked immutable.
const MUH: Meh = Meh {
    //~^ ERROR `UnsafeCell` in read-only memory
    x: &UnsafeCell::new(42),
};

struct Synced {
    x: UnsafeCell<i32>,
}
unsafe impl Sync for Synced {}

// Make sure we also catch this behind a type-erased `dyn Trait` reference.
const SNEAKY: &dyn Sync = &Synced { x: UnsafeCell::new(42) };
//~^ ERROR: `UnsafeCell` in read-only memory

// # Check for mutable references to read-only memory

static READONLY: i32 = 0;
static mut MUT_TO_READONLY: &mut i32 = unsafe { &mut *(&READONLY as *const _ as *mut _) };
//~^ ERROR: pointing to read-only memory

// # Check for consts pointing to mutable memory

static mut MUTABLE: i32 = 42;
const POINTS_TO_MUTABLE: &i32 = unsafe { &MUTABLE }; // OK, as long as it is not used as a pattern.

// This fails since `&*MUTABLE_REF` is basically a copy of `MUTABLE_REF`, but we
// can't read from that static as it is mutable.
static mut MUTABLE_REF: &mut i32 = &mut 42;
const POINTS_TO_MUTABLE2: &i32 = unsafe { &*MUTABLE_REF };
//~^ ERROR accesses mutable global memory

const POINTS_TO_MUTABLE_INNER: *const i32 = &mut 42 as *mut _ as *const _;
//~^ ERROR mutable pointer in final value

const POINTS_TO_MUTABLE_INNER2: *const i32 = &mut 42 as *const _;
//~^ ERROR mutable pointer in final value

// This does *not* error since it uses a shared reference, and we have to ignore
// those. See <https://github.com/rust-lang/rust/pull/128543>.
const INTERIOR_MUTABLE_BEHIND_RAW: *mut i32 = &UnsafeCell::new(42) as *const _ as *mut _;

struct SyncPtr<T> {
    x: *const T,
}
unsafe impl<T> Sync for SyncPtr<T> {}

// These pass the lifetime checks because of the "tail expression" / "outer scope" rule. (This
// relies on `SyncPtr` being a curly brace struct.) However, we intern the inner memory as
// read-only, so ideally this should be rejected. Unfortunately, as explained in
// <https://github.com/rust-lang/rust/pull/128543>, we have to accept it.
// (Also see `static-no-inner-mut` for similar tests on `static`.)
const RAW_SYNC: SyncPtr<AtomicI32> = SyncPtr { x: &AtomicI32::new(42) };

// With mutable references at least, we can detect this and error.
const RAW_MUT_CAST: SyncPtr<i32> = SyncPtr { x: &mut 42 as *mut _ as *const _ };
//~^ ERROR mutable pointer in final value

const RAW_MUT_COERCE: SyncPtr<i32> = SyncPtr { x: &mut 0 };
//~^ ERROR mutable pointer in final value

fn main() {
    unsafe {
        *MEH.x.get() = 99;
    }
    *OH_YES = 99; //~ ERROR cannot assign to `*OH_YES`, as `OH_YES` is an immutable static item
}

//~? WARN skipping const checks
