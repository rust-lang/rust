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
const B: *mut i32 = &mut 4; //~ ERROR mutable borrows of temporaries

// Ok, no actual mutable allocation exists
const B2: Option<&mut i32> = None;

// Not ok, can't prove that no mutable allocation ends up in final value
const B3: Option<&mut i32> = Some(&mut 42); //~ ERROR mutable borrows of temporaries

const fn helper(x: &mut i32) -> Option<&mut i32> { Some(x) }
const B4: Option<&mut i32> = helper(&mut 42); //~ ERROR temporary value dropped while borrowed

// Not ok, since it points to read-only memory.
const IMMUT_MUT_REF: &mut u16 = unsafe { mem::transmute(&13) };
//~^ ERROR pointing to read-only memory
static IMMUT_MUT_REF_STATIC: &mut u16 = unsafe { mem::transmute(&13) };
//~^ ERROR pointing to read-only memory

// Ok, because no borrows of mutable data exist here, since the `{}` moves
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
//~^ ERROR mutable borrows of temporaries
static RAW_MUT_COERCE_S: SyncPtr<i32> = SyncPtr { x: &mut 0 };
//~^ ERROR mutable borrows of temporaries
const RAW_MUT_CAST_C: SyncPtr<i32> = SyncPtr { x : &mut 42 as *mut _ as *const _ };
//~^ ERROR mutable borrows of temporaries
const RAW_MUT_COERCE_C: SyncPtr<i32> = SyncPtr { x: &mut 0 };
//~^ ERROR mutable borrows of temporaries

// Various cases of dangling references.
fn dangling() {
    const fn helper_int2ptr() -> Option<&'static mut i32> { unsafe {
        // Undefined behaviour (integer as pointer), who doesn't love tests like this.
        Some(&mut *(42 as *mut i32))
    } }
    const INT2PTR: Option<&mut i32> = helper_int2ptr(); //~ ERROR encountered a dangling reference
    static INT2PTR_STATIC: Option<&mut i32> = helper_int2ptr(); //~ ERROR encountered a dangling reference

    const fn helper_dangling() -> Option<&'static mut i32> { unsafe {
        // Undefined behaviour (dangling pointer), who doesn't love tests like this.
        Some(&mut *(&mut 42 as *mut i32))
    } }
    const DANGLING: Option<&mut i32> = helper_dangling(); //~ ERROR dangling reference
    static DANGLING_STATIC: Option<&mut i32> = helper_dangling(); //~ ERROR dangling reference

}

// Allowed, because there is an explicit static mut.
static mut BUFFER: i32 = 42;
const fn ptr_to_buffer() -> Option<&'static mut i32> { unsafe {
    Some(&mut *std::ptr::addr_of_mut!(BUFFER))
} }
const MUT_TO_BUFFER: Option<&mut i32> = ptr_to_buffer();

// These are fine! Just statics pointing to mutable statics, nothing fundamentally wrong with this.
static MUT_STATIC: Option<&mut i32> = ptr_to_buffer();
static mut MUT_ARRAY: &mut [u8] = &mut [42];
static MUTEX: std::sync::Mutex<&mut [u8]> = std::sync::Mutex::new(unsafe { &mut *MUT_ARRAY });

fn main() {
    println!("{}", unsafe { *A });
    unsafe { *B = 4 } // Bad news

    unsafe {
        **FOO.0.get() = 99;
        assert_eq!(**FOO.0.get(), 99);
    }
}
