#![feature(min_type_alias_impl_trait)]
#![feature(impl_trait_in_bindings)]
#![allow(incomplete_features)]

// failure-status: 101

type OpaqueOutputImpl<'a> = impl Output<'a> + 'a;

trait Output<'a> {}

impl<'a> Output<'a> for &'a str {}

fn cool_fn<'a>(arg: &'a str) -> OpaqueOutputImpl<'a> {
    //~^ ERROR Non-defining use
    let out: OpaqueOutputImpl<'a> = arg;
    //~^ ERROR not a universal region
    arg
}

fn main() {
    let s = String::from("wassup");
    cool_fn(&s);
}
