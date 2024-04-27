#![warn(clippy::self_named_module_files)]

mod bad;
mod more;

fn main() {
    let _ = bad::Thing;
    let _ = more::foo::Foo;
    let _ = more::inner::Inner;
}
