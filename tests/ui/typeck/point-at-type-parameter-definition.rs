trait Trait {
    fn do_stuff(&self);
}

struct Hello;

impl Hello {
    fn method(&self) {}
}

impl<Hello> Trait for Vec<Hello> {
    fn do_stuff(&self) {
        self[0].method(); //~ ERROR no method named `method` found for type parameter `Hello` in the current scope
    }
}

fn main() {}
