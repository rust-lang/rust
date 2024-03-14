//@ compile-flags: -Zunleash-the-miri-inside-of-you
//@ normalize-stderr-test "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr-test "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"

#![deny(const_eval_mutable_ptr_in_final_value)]
use std::cell::UnsafeCell;

// a test demonstrating what things we could allow with a smarter const qualification

static FOO: &&mut u32 = &&mut 42;
//~^ ERROR encountered mutable pointer in final value of static
//~| WARNING this was previously accepted by the compiler

static BAR: &mut () = &mut ();
//~^ ERROR encountered mutable pointer in final value of static
//~| WARNING this was previously accepted by the compiler

struct Foo<T>(T);

static BOO: &mut Foo<()> = &mut Foo(());
//~^ ERROR encountered mutable pointer in final value of static
//~| WARNING this was previously accepted by the compiler

struct Meh {
    x: &'static UnsafeCell<i32>,
}
unsafe impl Sync for Meh {}
static MEH: Meh = Meh { x: &UnsafeCell::new(42) };
//~^ ERROR encountered mutable pointer in final value of static
//~| WARNING this was previously accepted by the compiler

static OH_YES: &mut i32 = &mut 42;
//~^ ERROR encountered mutable pointer in final value of static
//~| WARNING this was previously accepted by the compiler
//~| ERROR it is undefined behavior to use this value

fn main() {
    unsafe {
        *MEH.x.get() = 99;
    }
    *OH_YES = 99; //~ ERROR cannot assign to `*OH_YES`, as `OH_YES` is an immutable static item
}
