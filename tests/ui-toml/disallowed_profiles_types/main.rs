#![warn(clippy::disallowed_types)]
#![allow(dead_code)]

use std::rc::Rc; //~ ERROR: use of a disallowed type `std::rc::Rc`
use std::sync::Mutex;

struct Wrapper;

fn default_type() {
    let _value: Rc<i32> = todo!(); //~ ERROR: use of a disallowed type `std::rc::Rc`
}

#[clippy::disallowed_profile("forward_pass")]
fn forward_profile() {
    let _value: std::cell::RefCell<i32> = todo!(); //~ ERROR: use of a disallowed type `std::cell::RefCell` (profile: forward_pass)
}

#[clippy::disallowed_profile("export")]
fn export_profile() {
    let _value: Mutex<i32> = todo!(); //~ ERROR: use of a disallowed type `std::sync::Mutex` (profile: export)
}

#[clippy::disallowed_profile("unknown_type_profile")]
//~^ ERROR: unknown profile `unknown_type_profile` for
//~| ERROR: unknown profile `unknown_type_profile` for
fn unknown_profile() {
    let _other = 1u32;
    let _fallback: Rc<i32> = todo!(); //~ ERROR: use of a disallowed type `std::rc::Rc`
}

#[clippy::disallowed_profiles("forward_pass", "export")]
fn merged_profiles() {
    let _value: std::cell::RefCell<i32> = todo!(); //~ ERROR: use of a disallowed type `std::cell::RefCell` (profile: forward_pass)
    let _other: Mutex<i32> = todo!(); //~ ERROR: use of a disallowed type `std::sync::Mutex` (profile: export)
}

fn main() {
    default_type();
    forward_profile();
    export_profile();
    unknown_profile();
    merged_profiles();
}
