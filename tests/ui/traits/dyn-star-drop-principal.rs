#![feature(dyn_star)]
#![allow(incomplete_features)]

trait Trait {}
impl Trait for usize {}

fn main() {
    // We allow &dyn Trait + Send -> &dyn Send (i.e. dropping principal),
    // but we don't (currently?) allow the same for dyn*
    let x: dyn* Trait + Send = 1usize;
    x as dyn* Send; //~ error: `dyn* Trait + Send` needs to have the same ABI as a pointer
}
