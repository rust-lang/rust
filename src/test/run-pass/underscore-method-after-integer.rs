trait Tr : Sized {
    fn _method_on_numbers(self) {}
}

impl Tr for i32 {}

fn main() {
    42._method_on_numbers();
}
