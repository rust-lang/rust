// `Rc` is not ever `Copy`, we should not suggest adding `T: Copy` constraint
fn duplicate_rc<T>(t: std::rc::Rc<T>) -> (std::rc::Rc<T>, std::rc::Rc<T>) {
    (t, t) //~ ERROR the type `Rc` does not implement `Copy`
}

fn main() {}
