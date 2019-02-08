fn foo(x: isize, ...) {
    //~^ ERROR: only foreign functions are allowed to be C-variadic
}

fn main() {}
