trait A {}

struct Struct {
    r: dyn A + 'static
}

fn new_struct(
    r: dyn A + 'static //~ ERROR the size for values of type
) -> Struct {
    Struct { r: r }
}

fn main() {}
