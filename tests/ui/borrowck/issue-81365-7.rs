use std::ops::Deref;

struct DerefTarget {
    target_field: bool,
}
struct Container {
    target: DerefTarget,
    container_field: bool,
}

impl Deref for Container {
    type Target = DerefTarget;
    fn deref(&self) -> &Self::Target {
        &self.target
    }
}

fn bad_borrow(c: &mut Container) {
    let first = &c.target_field;
    c.container_field = true; //~ ERROR E0506
    first;
}

fn main() {}
