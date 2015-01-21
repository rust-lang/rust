// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub static X: &'static str = "foobarbaz";
pub static Y: &'static [u8] = include_bytes!("lib.rs");

trait Foo {}
impl Foo for uint {}

pub fn dummy() {
    // force the vtable to be created
    let _x = &1u as &Foo;
}
