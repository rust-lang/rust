// https://github.com/rust-lang/rust/issues/5883
trait A {}

struct Struct {
    r: dyn A + 'static
}

fn new_struct(
    r: dyn A + 'static //~ ERROR the size for values of type
) -> Struct { //~ ERROR the size for values of type
    Struct { r: r }
}

fn main() {}
