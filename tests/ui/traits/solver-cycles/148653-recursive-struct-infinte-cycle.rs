// Issue #148653.
struct Vec<T> { //~ ERROR recursive type `Vec` has infinite size
    data: Vec<T>, //~ ERROR type parameter `T` is only used recursively
}
impl<T> Vec<T> {
    pub fn push(&mut self) -> &mut Self {
        self
    }
}

fn main() {}
