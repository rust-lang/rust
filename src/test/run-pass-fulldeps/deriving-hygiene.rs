#![allow(non_upper_case_globals)]
#![feature(rustc_private)]
extern crate serialize;
use serialize as rustc_serialize;

pub const other: u8 = 1;
pub const f: u8 = 1;
pub const d: u8 = 1;
pub const s: u8 = 1;
pub const state: u8 = 1;
pub const cmp: u8 = 1;

#[derive(Ord,Eq,PartialOrd,PartialEq,Debug,RustcDecodable,RustcEncodable,Hash)]
struct Foo {}

fn main() {
}
