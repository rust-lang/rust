use std::ops::{Deref, DerefMut};

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

impl DerefMut for Container {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.target
    }
}

impl Container {
    fn bad_borrow(&mut self) {
        let first = &mut self.target_field;
        self.container_field = true; //~ ERROR E0506
        first;
    }
}

fn main() {}
