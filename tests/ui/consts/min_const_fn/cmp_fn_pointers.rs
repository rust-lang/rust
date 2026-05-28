const fn cmp(x: fn(), y: fn()) -> bool {
    unsafe { x == y }
    //~^ ERROR pointers cannot
}

fn main() {}
