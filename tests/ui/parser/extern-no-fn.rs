extern "C" {
    f();
    //~^ ERROR missing `fn` or `struct` for function or struct definition
    //~| HELP if you meant to call a macro, try
}

fn main() {
}
