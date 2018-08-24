trait HasNumber<T> {
    const Number: usize;
}

enum One {}
enum Two {}

enum Foo {}

impl<T> HasNumber<T> for One {
    const Number: usize = 1;
}

impl<T> HasNumber<T> for Two {
    const Number: usize = 2;
}

fn main() {}
