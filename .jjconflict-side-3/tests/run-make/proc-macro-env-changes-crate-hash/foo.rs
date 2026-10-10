#![crate_type = "rlib"]

extern crate changing_macro;

changing_macro::emit_token!();

pub fn get() -> &'static str {
    TOKEN
}
