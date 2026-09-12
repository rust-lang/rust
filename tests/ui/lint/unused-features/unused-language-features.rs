#![crate_type = "lib"]
#![deny(unused_features)]
// Unused language features
#![feature(coroutines)]
//~^ ERROR feature `coroutines` is declared but not used
#![feature(coroutine_clone)]
//~^ ERROR feature `coroutine_clone` is declared but not used
#![feature(stmt_expr_attributes)]
//~^ ERROR feature `stmt_expr_attributes` is declared but not used
#![feature(asm_unwind)]
//~^ ERROR feature `asm_unwind` is declared but not used

// Enabled via cfg_attr, unused
#![cfg_attr(all(), feature(negative_impls))]
//~^ ERROR feature `negative_impls` is declared but not used

// Not enabled via cfg_attr, so should not warn even if unused
#![cfg_attr(any(), feature(type_ascription))]

macro_rules! use_asm_unwind {
    () => {
        unsafe { std::arch::asm!("", options(may_unwind)) };
    };
}
