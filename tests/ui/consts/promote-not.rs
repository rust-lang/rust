// ignore-tidy-linelength
// Test various things that we do not want to promote.
#![allow(unconditional_panic)]

use std::cell::Cell;
use std::mem::ManuallyDrop;

// We do not promote mutable references.
static mut TEST1: Option<&mut [i32]> = Some(&mut [1, 2, 3]); //~ ERROR temporary value dropped while borrowed

static mut TEST2: &'static mut [i32] = {
    let x = &mut [1,2,3]; //~ ERROR temporary value dropped while borrowed
    x
};

// We do not promote fn calls in `fn`, including `const fn`.
pub const fn promote_cal(b: bool) -> i32 {
    const fn foo() { [()][42] }

    if b {
        let _x: &'static () = &foo(); //~ ERROR temporary value dropped while borrowed
    }
    13
}

// We do not promote union field accesses in `fn`.
union U { x: i32, y: i32 }
pub const fn promote_union() {
    let _x: &'static i32 = &unsafe { U { x: 0 }.x }; //~ ERROR temporary value dropped while borrowed
}

// We do not promote union field accesses in `const`, either.
const TEST_UNION: () = {
    let _x: &'static i32 = &unsafe { U { x: 0 }.x }; //~ ERROR temporary value dropped while borrowed
};

// In a `const`, we do not promote things with interior mutability. Not even if we "project it away".
const TEST_INTERIOR_MUT: () = {
    // The "0." case is already ruled out by not permitting any interior mutability in `const`.
    let _val: &'static _ = &(Cell::new(1), 2).1; //~ ERROR temporary value dropped while borrowed
};

// This gets accepted by the "outer scope" rule, not promotion.
const TEST_DROP_OUTER_SCOPE: &String = &String::new();
// To demonstrate that, we can rewrite it as follows. If this was promotion it would still work.
const TEST_DROP_NOT_PROMOTE: &String = {
    let x = &String::new(); //~ ERROR destructor of `String` cannot be evaluated at compile-time
    // The "dropped while borrowed" error seems to be suppressed, but the key point is that this
    // fails to compile.
    x
};


// We do not promote function calls in `const` initializers in dead code.
const fn mk_panic() -> u32 { panic!() }
const fn mk_false() -> bool { false }
const Y: () = {
    if mk_false() {
        let _x: &'static u32 = &mk_panic(); //~ ERROR temporary value dropped while borrowed
    }
};

fn main() {
    // We must not promote things with interior mutability. Not even if we "project it away".
    let _val: &'static _ = &(Cell::new(1), 2).0; //~ ERROR temporary value dropped while borrowed
    let _val: &'static _ = &(Cell::new(1), 2).1; //~ ERROR temporary value dropped while borrowed

    // No promotion of fallible operations.
    let _val: &'static _ = &(1/0); //~ ERROR temporary value dropped while borrowed
    let _val: &'static _ = &(1/(1-1)); //~ ERROR temporary value dropped while borrowed
    let _val: &'static _ = &((1+1)/(1-1)); //~ ERROR temporary value dropped while borrowed
    let _val: &'static _ = &(i32::MIN/-1); //~ ERROR temporary value dropped while borrowed
    let _val: &'static _ = &(i32::MIN/(0-1)); //~ ERROR temporary value dropped while borrowed
    let _val: &'static _ = &(-128i8/-1); //~ ERROR temporary value dropped while borrowed
    let _val: &'static _ = &(1%0); //~ ERROR temporary value dropped while borrowed
    let _val: &'static _ = &(1%(1-1)); //~ ERROR temporary value dropped while borrowed
    let _val: &'static _ = &([1,2,3][4]+1); //~ ERROR temporary value dropped while borrowed

    // No promotion of temporaries that need to be dropped.
    const TEST_DROP: String = String::new();
    let _val: &'static _ = &TEST_DROP;
    //~^ ERROR temporary value dropped while borrowed
    let _val: &'static _ = &&TEST_DROP;
    //~^ ERROR temporary value dropped while borrowed
    //~| ERROR temporary value dropped while borrowed
    let _val: &'static _ = &(&TEST_DROP,);
    //~^ ERROR temporary value dropped while borrowed
    //~| ERROR temporary value dropped while borrowed
    let _val: &'static _ = &[&TEST_DROP; 1];
    //~^ ERROR temporary value dropped while borrowed
    //~| ERROR temporary value dropped while borrowed

    // Make sure there is no value-based reasoning for unions.
    union UnionWithCell {
        f1: i32,
        f2: ManuallyDrop<Cell<i32>>,
    }
    let x: &'static _ = &UnionWithCell { f1: 0 };
    //~^ ERROR temporary value dropped while borrowed
}
