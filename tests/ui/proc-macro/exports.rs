//@ force-host
//@ no-prefer-dynamic

#![crate_type = "proc-macro"]
#![allow(warnings)]

pub fn a() {} //~ ERROR: cannot export any items
pub struct B; //~ ERROR: cannot export any items
pub enum C {} //~ ERROR: cannot export any items
pub mod d {} //~ ERROR: cannot export any items

mod e {}
struct F;
enum G {}
fn h() {}
