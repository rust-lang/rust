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

fn main() {}
