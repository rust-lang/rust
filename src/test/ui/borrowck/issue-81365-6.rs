use std::ops::Deref;

struct Container {
    target: Vec<()>,
    container_field: bool,
}

impl Deref for Container {
    type Target = [()];
    fn deref(&self) -> &Self::Target {
        &self.target
    }
}

impl Container {
    fn bad_borrow(&mut self) {
        let first = &self[0];
        self.container_field = true; //~ ERROR E0506
        first;
    }
}

fn main() {}
