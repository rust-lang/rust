#![warn(clippy::disallowed_types)]

extern crate quote;
extern crate syn;

use std::sync as foo;
use std::sync::atomic::AtomicU32;
use std::time::Instant as Sneaky;

struct HashMap;

fn bad_return_type() -> fn() -> Sneaky {
    todo!()
}

fn bad_arg_type(_: impl Fn(Sneaky) -> foo::atomic::AtomicU32) {}

fn trait_obj(_: &dyn std::io::Read) {}

fn full_and_single_path_prim(_: usize, _: bool) {}

fn const_generics<const C: usize>() {}

struct GenArg<const U: usize>([u8; U]);

static BAD: foo::atomic::AtomicPtr<()> = foo::atomic::AtomicPtr::new(std::ptr::null_mut());

fn ip(_: std::net::Ipv4Addr) {}

fn listener(_: std::net::TcpListener) {}

#[allow(clippy::diverging_sub_expression)]
fn main() {
    let _: std::collections::HashMap<(), ()> = std::collections::HashMap::new();
    let _ = Sneaky::now();
    let _ = foo::atomic::AtomicU32::new(0);
    static FOO: std::sync::atomic::AtomicU32 = foo::atomic::AtomicU32::new(1);
    let _: std::collections::BTreeMap<(), syn::TypePath> = Default::default();
    let _ = syn::Ident::new("", todo!());
    let _ = HashMap;
    let _: usize = 64_usize;
}
