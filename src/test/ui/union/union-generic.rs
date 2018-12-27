use std::rc::Rc;

union U<T: Copy> {
    a: T
}

fn main() {
    let u = U { a: Rc::new(0u32) };
    //~^ ERROR  the trait bound `std::rc::Rc<u32>: std::marker::Copy` is not satisfied
    let u = U::<Rc<u32>> { a: Default::default() };
    //~^ ERROR  the trait bound `std::rc::Rc<u32>: std::marker::Copy` is not satisfied
}
