#![feature(rustc_attrs)]
#![feature(type_alias_impl_trait)]

mod foo {
    pub type Foo = impl Fn() -> usize;
    pub const fn bar() -> Foo {
        || 0usize
    }
}
use foo::*;
const BAZR: Foo = bar();

#[rustc_error]
fn main() {} //~ ERROR
