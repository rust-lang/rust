mod a;

mod b;
#[path = "b.rs"]
mod b2;

mod c;
#[path = "c.rs"]
mod c2;
#[path = "c.rs"]
mod c3;

mod from_other_module;
mod other_module;

mod d;
#[path = "d.rs"]
mod d2;
#[path = "d.rs"]
#[expect(clippy::duplicate_mod)]
mod d3;
#[path = "d.rs"]
#[allow(clippy::duplicate_mod)]
mod d4;

fn main() {}
