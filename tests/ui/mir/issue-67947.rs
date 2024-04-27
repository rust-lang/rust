struct Bug {
    A: [(); { *"" }.len()],
    //~^ ERROR: cannot move a value of type `str`
    //~| ERROR: cannot move out of a shared reference
}

fn main() {}
