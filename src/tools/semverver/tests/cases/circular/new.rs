use std::rc::Rc;

#[allow(dead_code)]
struct Abc {
    member: Option<Rc<Abc>>,
}
