#![feature(adt_const_params)]
#![deny(clippy::same_functions_in_if_condition)]
// ifs_same_cond warning is different from `ifs_same_cond`.
// clippy::if_same_then_else, clippy::comparison_chain -- all empty blocks
#![allow(incomplete_features)]
#![allow(
    clippy::comparison_chain,
    clippy::if_same_then_else,
    clippy::ifs_same_cond,
    clippy::uninlined_format_args
)]

use std::marker::ConstParamTy;

fn function() -> bool {
    true
}

fn fn_arg(_arg: u8) -> bool {
    true
}

struct Struct;

impl Struct {
    fn method(&self) -> bool {
        true
    }
    fn method_arg(&self, _arg: u8) -> bool {
        true
    }
}

fn ifs_same_cond_fn() {
    let a = 0;
    let obj = Struct;

    if function() {
    } else if function() {
        //~^ ERROR: `if` has the same function call as a previous `if`
    }

    if fn_arg(a) {
    } else if fn_arg(a) {
        //~^ ERROR: `if` has the same function call as a previous `if`
    }

    if obj.method() {
    } else if obj.method() {
        //~^ ERROR: `if` has the same function call as a previous `if`
    }

    if obj.method_arg(a) {
    } else if obj.method_arg(a) {
        //~^ ERROR: `if` has the same function call as a previous `if`
    }

    let mut v = vec![1];
    if v.pop().is_none() {
    } else if v.pop().is_none() {
        //~^ ERROR: `if` has the same function call as a previous `if`
    }

    if v.len() == 42 {
    } else if v.len() == 42 {
        //~^ ERROR: `if` has the same function call as a previous `if`
    }

    if v.len() == 1 {
        // ok, different conditions
    } else if v.len() == 2 {
    }

    if fn_arg(0) {
        // ok, different arguments.
    } else if fn_arg(1) {
    }

    if obj.method_arg(0) {
        // ok, different arguments.
    } else if obj.method_arg(1) {
    }

    if a == 1 {
        // ok, warning is on `ifs_same_cond` behalf.
    } else if a == 1 {
    }
}

fn main() {
    // macro as condition (see #6168)
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    };
    println!("{}", os);

    #[derive(PartialEq, Eq, ConstParamTy)]
    enum E {
        A,
        B,
    }
    fn generic<const P: E>() -> bool {
        match P {
            E::A => true,
            E::B => false,
        }
    }
    if generic::<{ E::A }>() {
        println!("A");
    } else if generic::<{ E::B }>() {
        println!("B");
    }
}
