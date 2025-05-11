#![allow(dead_code, path_statements)]
#![deny(unused_attributes, unused_must_use)]
#![feature(asm_experimental_arch, stmt_expr_attributes, trait_alias)]

#[must_use] //~ ERROR `#[must_use]` has no effect
extern crate std as std2;

#[must_use] //~ ERROR `#[must_use]` has no effect
mod test_mod {}

#[must_use] //~ ERROR `#[must_use]` has no effect
use std::arch::global_asm;

#[must_use] //~ ERROR `#[must_use]` has no effect
const CONST: usize = 4;
#[must_use] //~ ERROR `#[must_use]` has no effect
#[no_mangle]
static STATIC: usize = 4;

#[must_use]
struct X;

#[must_use]
enum Y {
    Z,
}

#[must_use]
union U {
    unit: (),
}

#[must_use] //~ ERROR `#[must_use]` has no effect
impl U {
    #[must_use]
    fn method() -> i32 {
        4
    }
}

#[must_use]
#[no_mangle]
fn foo() -> i64 {
    4
}

#[must_use] //~ ERROR `#[must_use]` has no effect
extern "Rust" {
    #[link_name = "STATIC"]
    #[must_use] //~ ERROR `#[must_use]` has no effect
    static FOREIGN_STATIC: usize;

    #[link_name = "foo"]
    #[must_use]
    fn foreign_foo() -> i64;
}

#[must_use] //~ ERROR unused attribute
global_asm!("");

#[must_use] //~ ERROR `#[must_use]` has no effect
type UseMe = ();

fn qux<#[must_use] T>(_: T) {} //~ ERROR `#[must_use]` has no effect

#[must_use]
trait Use {
    #[must_use] //~ ERROR `#[must_use]` has no effect
    const ASSOC_CONST: usize = 4;
    #[must_use] //~ ERROR `#[must_use]` has no effect
    type AssocTy;

    #[must_use]
    fn get_four(&self) -> usize {
        4
    }
}

#[must_use] //~ ERROR `#[must_use]` has no effect
impl Use for () {
    type AssocTy = ();

    #[must_use] //~ ERROR `#[must_use]` has no effect
    fn get_four(&self) -> usize {
        4
    }
}

#[must_use] //~ ERROR `#[must_use]` has no effect
trait Alias = Use;

#[must_use] //~ ERROR `#[must_use]` has no effect
macro_rules! cool_macro {
    () => {
        4
    };
}

fn main() {
    #[must_use] //~ ERROR `#[must_use]` has no effect
    let x = || {};
    x();

    let x = #[must_use] //~ ERROR `#[must_use]` has no effect
    || {};
    x();

    X; //~ ERROR that must be used
    Y::Z; //~ ERROR that must be used
    U { unit: () }; //~ ERROR that must be used
    U::method(); //~ ERROR that must be used
    foo(); //~ ERROR that must be used

    unsafe {
        foreign_foo(); //~ ERROR that must be used
    };

    CONST;
    STATIC;
    unsafe { FOREIGN_STATIC };
    cool_macro!();
    qux(4);
    ().get_four(); //~ ERROR that must be used

    match Some(4) {
        #[must_use] //~ ERROR `#[must_use]` has no effect
        Some(res) => res,
        None => 0,
    };

    struct PatternField {
        foo: i32,
    }
    let s = PatternField { #[must_use]  foo: 123 }; //~ ERROR `#[must_use]` has no effect
    let PatternField { #[must_use] foo } = s; //~ ERROR `#[must_use]` has no effect
}
