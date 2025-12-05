//@ run-rustfix

#![allow(dead_code, path_statements)]
#![deny(unused_attributes, unused_must_use)]
#![feature(asm_experimental_arch, stmt_expr_attributes, trait_alias)]

#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
extern crate std as std2;

#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
mod test_mod {}

#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
use std::arch::global_asm;

#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
const CONST: usize = 4;
#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
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

#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
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

#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
extern "Rust" {
    #[link_name = "STATIC"]
    #[must_use] //~ ERROR attribute cannot be used on
    //~| WARN previously accepted
    static FOREIGN_STATIC: usize;

    #[link_name = "foo"]
    #[must_use]
    fn foreign_foo() -> i64;
}

#[must_use]
//~^ ERROR `#[must_use]` attribute cannot be used on macro calls
//~| WARN this was previously accepted by the compiler but is being phased out
global_asm!("");

#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
type UseMe = ();

fn qux<#[must_use] T>(_: T) {} //~ ERROR attribute cannot be used on
//~| WARN previously accepted

#[must_use]
trait Use {
    #[must_use] //~ ERROR attribute cannot be used on
    //~| WARN previously accepted
    const ASSOC_CONST: usize = 4;
    #[must_use] //~ ERROR attribute cannot be used on
    //~| WARN previously accepted
    type AssocTy;

    #[must_use]
    fn get_four(&self) -> usize {
        4
    }
}

#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
impl Use for () {
    type AssocTy = ();

    #[must_use] //~ ERROR attribute cannot be used on
    //~| WARN previously accepted
    fn get_four(&self) -> usize {
        4
    }
}

#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
trait Alias = Use;

#[must_use] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
macro_rules! cool_macro {
    () => {
        4
    };
}

fn main() {
    #[must_use] //~ ERROR attribute cannot be used on
    //~| WARN previously accepted
    let x = || {};
    x();

    let x = #[must_use] //~ ERROR attribute cannot be used on
    //~| WARN previously accepted
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
        #[must_use] //~ ERROR attribute cannot be used on
        //~| WARN previously accepted
        Some(res) => res,
        None => 0,
    };

    struct PatternField {
        foo: i32,
    }
    let s = PatternField { #[must_use]  foo: 123 }; //~ ERROR attribute cannot be used on
    //~| WARN previously accepted
    let PatternField { #[must_use] foo } = s; //~ ERROR attribute cannot be used on
    //~| WARN previously accepted
    let _ = foo;
}
