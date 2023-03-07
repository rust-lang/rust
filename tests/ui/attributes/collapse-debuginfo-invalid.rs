#![feature(collapse_debuginfo)]
#![feature(stmt_expr_attributes)]
#![feature(type_alias_impl_trait)]
#![no_std]

// Test that the `#[collapse_debuginfo]` attribute can only be used on macro definitions.

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
extern crate std;

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
use std::collections::HashMap;

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
static FOO: u32 = 3;

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
const BAR: u32 = 3;

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
fn foo() {
    let _ = #[collapse_debuginfo] || { };
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
    #[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
    let _ = 3;
    let _ = #[collapse_debuginfo] 3;
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
    match (3, 4) {
        #[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
        _ => (),
    }
}

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
mod bar {
}

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
type Map = HashMap<u32, u32>;

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
enum Foo {
    #[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
    Variant,
}

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
struct Bar {
    #[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
    field: u32,
}

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
union Qux {
    a: u32,
    b: u16
}

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
trait Foobar {
    #[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
    type Bar;
}

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
type AFoobar = impl Foobar;

impl Foobar for Bar {
    type Bar = u32;
}

fn constraining() -> AFoobar {
    Bar { field: 3 }
}

#[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
impl Bar {
    #[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
    const FOO: u32 = 3;

    #[collapse_debuginfo]
//~^ ERROR `collapse_debuginfo` attribute should be applied to macro definitions
    fn bar(&self) {}
}

#[collapse_debuginfo]
macro_rules! finally {
    ($e:expr) => { $e }
}

fn main() {}
