struct Struct {
    y: (typeof("hey"),),
    //~^ ERROR `typeof` is a reserved keyword but unimplemented
}

fn main() {}
