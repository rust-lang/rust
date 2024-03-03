#![warn(clippy::thread_local_initializer_can_be_made_const)]

use std::cell::RefCell;

fn main() {
    // lint and suggest const
    thread_local! {
        static BUF_1: RefCell<String> = RefCell::new(String::new());
    }
    //~^^ ERROR: initializer for `thread_local` value can be made `const`

    // don't lint
    thread_local! {
        static BUF_2: RefCell<String> = const { RefCell::new(String::new()) };
    }

    thread_local! {
        static SIMPLE:i32 = 1;
    }
    //~^^ ERROR: initializer for `thread_local` value can be made `const`

    // lint and suggest const for all non const items
    thread_local! {
        static BUF_3_CAN_BE_MADE_CONST: RefCell<String> = RefCell::new(String::new());
        static CONST_MIXED_WITH:i32 = const { 1 };
        static BUF_4_CAN_BE_MADE_CONST: RefCell<String> = RefCell::new(String::new());
    }
    //~^^^^ ERROR: initializer for `thread_local` value can be made `const`
    //~^^^ ERROR: initializer for `thread_local` value can be made `const`

    thread_local! {
        static PEEL_ME: i32 = { 1 };
        //~^ ERROR: initializer for `thread_local` value can be made `const`
        static PEEL_ME_MANY: i32 = { let x = 1; x * x };
        //~^ ERROR: initializer for `thread_local` value can be made `const`
    }
}

#[clippy::msrv = "1.58"]
fn f() {
    thread_local! {
        static TLS: i32 = 1;
    }
}
