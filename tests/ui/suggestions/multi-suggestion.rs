//@ revisions: ascii unicode
//@[unicode] compile-flags: -Zunstable-options --error-format=human-unicode

#![allow(dead_code)]
struct U <T> {
    wtf: Option<Box<U<T>>>,
    x: T,
}
fn main() {
    U {
        wtf: Some(Box(U { //[ascii]~ ERROR cannot initialize a tuple struct which contains private fields
            wtf: None,
            x: (),
        })),
        x: ()
    };
    let _ = std::collections::HashMap();
    //[ascii]~^ ERROR expected function, tuple struct or tuple variant, found struct `std::collections::HashMap`
    let _ = std::collections::HashMap {};
    //[ascii]~^ ERROR cannot construct `HashMap<_, _, _>` with struct literal syntax due to private fields
    let _ = Box {}; //[ascii]~ ERROR cannot construct `Box<_, _>` with struct literal syntax due to private fields
}
