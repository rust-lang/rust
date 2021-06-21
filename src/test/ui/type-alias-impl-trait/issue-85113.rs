#![feature(min_type_alias_impl_trait)]
#![feature(impl_trait_in_bindings)]
#![allow(incomplete_features)]

type OpaqueOutputImpl<'a> = impl Output<'a> + 'a;
//~^ ERROR: hidden type for `impl Trait` captures lifetime that does not appear in bounds
//~| ERROR: the type `&'<empty> str` does not fulfill the required lifetime
//~| ERROR: cannot infer an appropriate lifetime for lifetime parameter `'a` due to conflicting requirements

trait Output<'a> {}

impl<'a> Output<'a> for &'a str {}

fn cool_fn<'a>(arg: &'a str) -> OpaqueOutputImpl<'a> {
    let out: OpaqueOutputImpl<'a> = arg;
    arg
}

fn main() {
    let s = String::from("wassup");
    cool_fn(&s);
}
