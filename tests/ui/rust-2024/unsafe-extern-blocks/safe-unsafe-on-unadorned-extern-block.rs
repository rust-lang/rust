extern "C" {
    safe fn test1(i: i32);
    //~^ ERROR items in unadorned `extern` blocks cannot have safety qualifiers
}

fn test2(i: i32) {
    test1(i);
}

fn main() {}
