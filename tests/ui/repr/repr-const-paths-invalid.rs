#![feature(const_attr_paths)]

fn not_const() -> usize {
    8
}

#[repr(align(not_const))]
//~^ ERROR paths in `repr(align)` must refer to `const` items, not function
struct WrongItem;

const BOOL_ALIGN: bool = true;
#[repr(align(BOOL_ALIGN))]
//~^ ERROR const item used in `repr(align)` must have an integer type
struct WrongType;

const NEG_ALIGN: isize = -8;
#[repr(align(NEG_ALIGN))]
//~^ ERROR const item used in `repr(align)` must evaluate to a non-negative integer
struct Negative;

#[repr(align(N))]
//~^ ERROR cannot find value `N` in this scope
struct Generic<const N: usize>;

const CYCLE: usize = core::mem::align_of::<Cycle>();
#[repr(align(CYCLE))]
struct Cycle; //~ ERROR cycle detected when computing ADT definition for `Cycle`

fn main() {}
