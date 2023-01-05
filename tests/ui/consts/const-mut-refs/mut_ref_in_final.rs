#![feature(const_mut_refs)]
#![feature(raw_ref_op)]

const NULL: *mut i32 = std::ptr::null_mut();
const A: *const i32 = &4;

// It could be made sound to allow it to compile,
// but we do not want to allow this to compile,
// as that would be an enormous footgun in oli-obk's opinion.
const B: *mut i32 = &mut 4; //~ ERROR mutable references are not allowed

// Ok, no actual mutable allocation exists
const B2: Option<&mut i32> = None;

// Not ok, can't prove that no mutable allocation ends up in final value
const B3: Option<&mut i32> = Some(&mut 42); //~ ERROR temporary value dropped while borrowed

const fn helper(x: &mut i32) -> Option<&mut i32> { Some(x) }
const B4: Option<&mut i32> = helper(&mut 42); //~ ERROR temporary value dropped while borrowed

// Ok, because no references to mutable data exist here, since the `{}` moves
// its value and then takes a reference to that.
const C: *const i32 = &{
    let mut x = 42;
    x += 3;
    x
};

use std::cell::UnsafeCell;
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

fn main() {
    println!("{}", unsafe { *A });
    unsafe { *B = 4 } // Bad news

    unsafe {
        **FOO.0.get() = 99;
        assert_eq!(**FOO.0.get(), 99);
    }
}
