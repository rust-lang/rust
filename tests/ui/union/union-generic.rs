use std::rc::Rc;

union U<T: Copy> {
    a: T
}

fn main() {
    let u = U { a: Rc::new(0u32) };
    //~^ ERROR trait `Copy` is not implemented for `Rc<u32>`
    let u = U::<Rc<u32>> { a: Default::default() };
    //~^ ERROR trait `Copy` is not implemented for `Rc<u32>`
}
