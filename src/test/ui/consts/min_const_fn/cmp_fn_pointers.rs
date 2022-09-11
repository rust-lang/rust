const fn cmp(x: fn(), y: fn()) -> bool {
    unsafe { x == y }
    //~^ ERROR can't compare
}

fn main() {}
