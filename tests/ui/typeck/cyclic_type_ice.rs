fn thing() {
    let f = |_, _| ();
    f(f); //~ ERROR: closure/coroutine type that references itself
    //~^ ERROR: this function takes 2 arguments but 1 argument was supplied
}

fn main() {}
