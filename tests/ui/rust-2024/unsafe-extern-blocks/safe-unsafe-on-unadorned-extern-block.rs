extern "C" {
    safe static TEST1: i32;
    //~^ ERROR items in unadorned `extern` blocks cannot have safety qualifiers
    safe fn test1(i: i32);
    //~^ ERROR items in unadorned `extern` blocks cannot have safety qualifiers
}

fn test2() {
    test1(TEST1);
}

fn main() {}
