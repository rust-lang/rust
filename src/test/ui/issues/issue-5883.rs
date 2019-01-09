trait A {}

struct Struct {
    r: A+'static
}

fn new_struct(r: A+'static)
    -> Struct { //~^ ERROR the size for values of type
    //~^ ERROR the size for values of type
    Struct { r: r }
}

fn main() {}
