#![feature(inline_const_pat)]

fn main() {
    match () {
        const { (|| {})() } => {}
        //~^ ERROR cannot call non-const closure in constants
    }
}
