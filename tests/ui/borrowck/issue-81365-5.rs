use std::ops::Deref;

struct DerefTarget {
    target_field: bool,
}

impl DerefTarget {
    fn get(&self) -> &bool {
        &self.target_field
    }
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

impl Container {
    fn bad_borrow(&mut self) {
        let first = self.get();
        self.container_field = true; //~ ERROR E0506
        first;
    }
}

fn main() {}
