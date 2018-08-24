#![feature(rustc_private)]
extern crate serialize;

pub const other: u8 = 1;
pub const f: u8 = 1;
pub const d: u8 = 1;
pub const s: u8 = 1;
pub const state: u8 = 1;
pub const cmp: u8 = 1;

#[derive(Ord,Eq,PartialOrd,PartialEq,Debug,Decodable,Encodable,Hash)]
struct Foo {}

fn main() {
}
