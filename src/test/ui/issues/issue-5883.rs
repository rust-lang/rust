trait A {}

struct Struct {
    r: dyn A + 'static
}

fn new_struct(r: dyn A + 'static)
    -> Struct { //~^ ERROR the size for values of type
    //~^ ERROR the size for values of type
    Struct { r: r }
}

fn main() {}
