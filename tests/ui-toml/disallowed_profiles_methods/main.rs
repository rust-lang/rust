#![warn(clippy::disallowed_methods)]
#![allow(
    unused,
    clippy::no_effect,
    clippy::needless_borrow,
    clippy::vec_init_then_push,
    clippy::unnecessary_literal_unwrap
)]

fn default_violation() {
    let value = String::from("test");
    std::mem::drop(value); //~ ERROR: use of a disallowed method `std::mem::drop`
}

#[expect(clippy::disallowed_methods)]
fn expected_violation() {
    let value = String::from("test");
    std::mem::drop(value);
}

#[clippy::disallowed_profile("forward_pass")]
fn forward_profile() {
    let mut values = Vec::new();
    values.push(1); //~ ERROR: use of a disallowed method `alloc::vec::Vec::push` (profile: forward_pass)
}

#[clippy::disallowed_profile("export")]
fn export_profile() {
    let value = Some(1);
    value.unwrap(); //~ ERROR: use of a disallowed method `core::option::Option::unwrap` (profile: export)
}

#[clippy::disallowed_profile("unknown_profile")]
//~^ ERROR: unknown profile `unknown_profile` for
//~| ERROR: unknown profile `unknown_profile` for
fn unknown_profile() {
    let mut values = Vec::new();
    values.push(1);
    // unknown profile falls back to the default list
    std::mem::drop(values); //~ ERROR: use of a disallowed method `std::mem::drop`
}

#[clippy::disallowed_profiles("forward_pass", "export")]
fn merged_profiles() {
    let mut values = Vec::new();
    values.push(1); //~ ERROR: use of a disallowed method `alloc::vec::Vec::push` (profile: forward_pass)
    let value = Some(1);
    value.unwrap(); //~ ERROR: use of a disallowed method `core::option::Option::unwrap` (profile: export)
}

fn main() {
    default_violation();
    expected_violation();
    forward_profile();
    export_profile();
    unknown_profile();
    merged_profiles();
}
