#![feature(on_unimplemented)]

#[rustc_on_unimplemented]
//~^ ERROR E0232
trait Bar {}

fn main() {
}
