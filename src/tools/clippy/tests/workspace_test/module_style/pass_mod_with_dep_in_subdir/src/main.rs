#![deny(clippy::self_named_module_files)]

mod bad;
mod more;
extern crate dep_no_mod;

fn main() {
    let _ = bad::Thing;
    let _ = more::foo::Foo;
    let _ = more::inner::Inner;
    let _ = dep_no_mod::foo::Thing;
    let _ = dep_no_mod::foo::hello::Hello;
}
