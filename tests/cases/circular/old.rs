use std::rc::Rc;

struct Abc {
    member: Option<Rc<Abc>>,
}
