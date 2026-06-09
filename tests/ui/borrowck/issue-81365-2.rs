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

struct Outer {
    container: Container,
}

impl Outer {
    fn bad_borrow(&mut self) {
        let first = &self.container.target_field;
        self.container.container_field = true; //~ ERROR E0506
        first;
    }
}

fn main() {}
