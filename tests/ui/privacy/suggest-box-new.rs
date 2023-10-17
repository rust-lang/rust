// run-rustfix
#![allow(dead_code)]
struct U <T> {
    wtf: Option<Box<U<T>>>,
    x: T,
}
fn main() {
    U {
        wtf: Some(Box(U { //~ ERROR cannot initialize a tuple struct which contains private fields
            wtf: None,
            x: (),
        })),
        x: ()
    };
}
