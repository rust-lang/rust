//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "( 0x[0-9a-f][0-9a-f] │)? ([0-9a-f][0-9a-f] |__ |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> " HEX_DUMP"
//@ normalize-stderr: "HEX_DUMP\s*\n\s*HEX_DUMP" -> "HEX_DUMP"
//@ dont-require-annotations: NOTE

use std::cell::UnsafeCell;
use std::mem;

const NULL: *mut i32 = std::ptr::null_mut();
const A: *const i32 = &4;

// It could be made sound to allow it to compile,
// but we do not want to allow this to compile,
// as that would be an enormous footgun in oli-obk's opinion.
const B: *mut i32 = &mut 4; //~ ERROR mutable references are not allowed

// Ok, no actual mutable allocation exists
const B2: Option<&mut i32> = None;

// Not ok, can't prove that no mutable allocation ends up in final value
const B3: Option<&mut i32> = Some(&mut 42); //~ ERROR mutable references are not allowed

const fn helper(x: &mut i32) -> Option<&mut i32> { Some(x) }
const B4: Option<&mut i32> = helper(&mut 42); //~ ERROR temporary value dropped while borrowed

// Not ok, since it points to read-only memory.
const IMMUT_MUT_REF: &mut u16 = unsafe { mem::transmute(&13) };
//~^ ERROR pointing to read-only memory

// Ok, because no references to mutable data exist here, since the `{}` moves
// its value and then takes a reference to that.
const C: *const i32 = &{
    let mut x = 42;
    x += 3;
    x
};

// Still ok, since `x` will be moved before the final pointer is crated,
// so `_ref` doesn't actually point to the memory that escapes.
const C_NO: *const i32 = &{
    let mut x = 42;
    let _ref = &mut x;
    x
};

struct NotAMutex<T>(UnsafeCell<T>);

unsafe impl<T> Sync for NotAMutex<T> {}

const FOO: NotAMutex<&mut i32> = NotAMutex(UnsafeCell::new(&mut 42));
//~^ ERROR temporary value dropped while borrowed

static FOO2: NotAMutex<&mut i32> = NotAMutex(UnsafeCell::new(&mut 42));
//~^ ERROR temporary value dropped while borrowed

static mut FOO3: NotAMutex<&mut i32> = NotAMutex(UnsafeCell::new(&mut 42));
//~^ ERROR temporary value dropped while borrowed

// `BAR` works, because `&42` promotes immediately instead of relying on
// the enclosing scope rule.
const BAR: NotAMutex<&i32> = NotAMutex(UnsafeCell::new(&42));

struct SyncPtr<T> { x : *const T }
unsafe impl<T> Sync for SyncPtr<T> {}

// These pass the lifetime checks because of the "tail expression" / "outer scope" rule.
// (This relies on `SyncPtr` being a curly brace struct.)
// However, we intern the inner memory as read-only, so this must be rejected.
static RAW_MUT_CAST_S: SyncPtr<i32> = SyncPtr { x : &mut 42 as *mut _ as *const _ };
//~^ ERROR mutable references are not allowed
static RAW_MUT_COERCE_S: SyncPtr<i32> = SyncPtr { x: &mut 0 };
//~^ ERROR mutable references are not allowed
const RAW_MUT_CAST_C: SyncPtr<i32> = SyncPtr { x : &mut 42 as *mut _ as *const _ };
//~^ ERROR mutable references are not allowed
const RAW_MUT_COERCE_C: SyncPtr<i32> = SyncPtr { x: &mut 0 };
//~^ ERROR mutable references are not allowed

fn main() {
    println!("{}", unsafe { *A });
    unsafe { *B = 4 } // Bad news

    unsafe {
        **FOO.0.get() = 99;
        assert_eq!(**FOO.0.get(), 99);
    }
}
