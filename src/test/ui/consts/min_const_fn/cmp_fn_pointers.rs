const fn cmp(x: fn(), y: fn()) -> bool {
    unsafe { x == y }
    //~^ ERROR pointers cannot be reliably compared
}

fn main() {}
