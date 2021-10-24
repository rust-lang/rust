#![allow(non_fmt_panics)]
#![feature(const_panic_extra)]
#![crate_type = "lib"]

const _: () = core::panic!("hello {}", "world");
//~^ ERROR evaluation of constant value failed

const _: () = core::panic!("answer is {ans}", ans = 21*2);
//~^ ERROR evaluation of constant value failed

const _: () = core::assert_eq!(42, 43);
//~^ ERROR evaluation of constant value failed

const _: () = core::assert_ne!(42, 42, "hello {}", "world");
//~^ ERROR evaluation of constant value failed

const _: () = core::panic!("{}", 42);
//~^ ERROR evaluation of constant value failed

const _: () = std::panic!(42);
//~^ ERROR evaluation of constant value failed

const _: () = std::panic!(String::new());
//~^ ERROR evaluation of constant value failed
