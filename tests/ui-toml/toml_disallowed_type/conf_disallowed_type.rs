#![warn(clippy::disallowed_type)]

extern crate quote;
extern crate syn;

use std::sync as foo;
use std::sync::atomic::AtomicU32;
use std::time::Instant as Sneaky;

struct HashMap;

fn bad_return_type() -> fn() -> Sneaky {
    todo!()
}

fn bad_arg_type(_: impl Fn(Sneaky) -> foo::atomic::AtomicU32) {
    todo!()
}

fn trait_obj(_: &dyn std::io::Read) {
    todo!()
}

static BAD: foo::atomic::AtomicPtr<()> = foo::atomic::AtomicPtr::new(std::ptr::null_mut());

#[allow(clippy::diverging_sub_expression)]
fn main() {
    let _: std::collections::HashMap<(), ()> = std::collections::HashMap::new();
    let _ = Sneaky::now();
    let _ = foo::atomic::AtomicU32::new(0);
    static FOO: std::sync::atomic::AtomicU32 = foo::atomic::AtomicU32::new(1);
    let _: std::collections::BTreeMap<(), syn::TypePath> = Default::default();
    let _ = syn::Ident::new("", todo!());
    let _ = HashMap;
}
