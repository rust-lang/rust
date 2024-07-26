// Checks to ensure that we properly detect when a closure constrains an opaque type

#![feature(type_alias_impl_trait, stmt_expr_attributes)]

use std::fmt::Debug;

fn main() {
    type Opaque = impl Debug;
    #[define_opaques(Opaque)]
    fn _unused() -> Opaque { String::new() }
    let null = #[define_opaques(Opaque)] || -> Opaque { 0 };
    //~^ ERROR: concrete type differs from previous defining opaque type use
    println!("{:?}", null());
}
