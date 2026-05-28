#![warn(clippy::disallowed_types)]

extern crate quote;
extern crate syn;

use std::sync as foo;
use std::sync::atomic::AtomicU32;
//~^ disallowed_types
use std::time::Instant as Sneaky;
//~^ disallowed_types

struct HashMap;

fn bad_return_type() -> fn() -> Sneaky {
    //~^ disallowed_types
    todo!()
}

fn bad_arg_type(_: impl Fn(Sneaky) -> foo::atomic::AtomicU32) {}
//~^ disallowed_types
//~| disallowed_types

fn trait_obj(_: &dyn std::io::Read) {}
//~^ disallowed_types

fn full_and_single_path_prim(_: usize, _: bool) {}
//~^ disallowed_types
//~| disallowed_types

fn const_generics<const C: usize>() {}
//~^ disallowed_types

struct GenArg<const U: usize>([u8; U]);
//~^ disallowed_types

static BAD: foo::atomic::AtomicPtr<()> = foo::atomic::AtomicPtr::new(std::ptr::null_mut());

fn ip(_: std::net::Ipv4Addr) {}
//~^ disallowed_types

fn listener(_: std::net::TcpListener) {}
//~^ disallowed_types

#[allow(clippy::diverging_sub_expression)]
fn main() {
    let _: std::collections::HashMap<(), ()> = std::collections::HashMap::new();
    //~^ disallowed_types
    //~| disallowed_types
    let _ = Sneaky::now();
    //~^ disallowed_types
    let _ = foo::atomic::AtomicU32::new(0);
    //~^ disallowed_types
    static FOO: std::sync::atomic::AtomicU32 = foo::atomic::AtomicU32::new(1);
    //~^ disallowed_types
    //~| disallowed_types
    let _: std::collections::BTreeMap<(), syn::TypePath> = Default::default();
    //~^ disallowed_types
    let _ = syn::Ident::new("", todo!());
    //~^ disallowed_types
    let _ = HashMap;
    let _: usize = 64_usize;
    //~^ disallowed_types
}

mod useless_attribute {
    // Regression test for https://github.com/rust-lang/rust-clippy/issues/12753
    #[allow(clippy::disallowed_types)]
    use std::collections::HashMap;
}
