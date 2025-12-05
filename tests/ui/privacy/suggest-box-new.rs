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
    let _ = std::collections::HashMap();
    //~^ ERROR expected function, tuple struct or tuple variant, found struct `std::collections::HashMap`
    let _ = std::collections::HashMap {};
    //~^ ERROR cannot construct `HashMap<_, _, _>` with struct literal syntax due to private fields
    let _ = Box {}; //~ ERROR cannot construct `Box<_, _>` with struct literal syntax due to private fields

    // test that we properly instantiate the parameter of `Box::<T>::new` with an inference variable
    let _ = Box::<i32> {}; //~ ERROR cannot construct `Box<i32>` with struct literal syntax due to private fields
}
