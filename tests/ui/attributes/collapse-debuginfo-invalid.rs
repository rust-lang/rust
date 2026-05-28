#![feature(stmt_expr_attributes)]
#![feature(type_alias_impl_trait)]
#![no_std]

// Test that the `#[collapse_debuginfo]` attribute can only be used on macro definitions.

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
extern crate std;

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
use std::collections::HashMap;

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
static FOO: u32 = 3;

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
const BAR: u32 = 3;

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
fn foo() {
    let _ = #[collapse_debuginfo(yes)] || { };
    //~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
    #[collapse_debuginfo(yes)]
    //~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
    let _ = 3;
    let _ = #[collapse_debuginfo(yes)] 3;
    //~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
    match (3, 4) {
        #[collapse_debuginfo(yes)]
        //~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
        _ => (),
    }
}

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
mod bar {
}

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
type Map = HashMap<u32, u32>;

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
enum Foo {
    #[collapse_debuginfo(yes)]
    //~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
    Variant,
}

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
struct Bar {
    #[collapse_debuginfo(yes)]
    //~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
    field: u32,
}

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
union Qux {
    a: u32,
    b: u16
}

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
trait Foobar {
    #[collapse_debuginfo(yes)]
    //~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
    type Bar;
}

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
type AFoobar = impl Foobar;

impl Foobar for Bar {
    type Bar = u32;
}

#[define_opaque(AFoobar)]
fn constraining() -> AFoobar {
    Bar { field: 3 }
}

#[collapse_debuginfo(yes)]
//~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
impl Bar {
    #[collapse_debuginfo(yes)]
    //~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
    const FOO: u32 = 3;

    #[collapse_debuginfo(yes)]
    //~^ ERROR `#[collapse_debuginfo]` attribute cannot be used on
    fn bar(&self) {}
}

#[collapse_debuginfo(yes)]
macro_rules! finally {
    ($e:expr) => { $e }
}

fn main() {}
