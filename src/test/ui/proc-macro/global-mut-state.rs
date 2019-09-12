// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![allow(warnings)]

use std::cell::Cell;
use std::sync::atomic::AtomicBool;

static mut FOO: u8 = 0;
//~^ ERROR mutable global state in a proc-macro

static BAR: AtomicBool = AtomicBool::new(false);
//~^ ERROR mutable global state in a proc-macro

thread_local!(static BAZ: Cell<String> = Cell::new(String::new()));
//~^ ERROR mutable global state in a proc-macro

static FROZEN: &str = "snow";
