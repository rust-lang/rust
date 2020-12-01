const fn cmp(x: fn(), y: fn()) -> bool {
    //~^ ERROR function pointer
    //~| ERROR function pointer
    unsafe { x == y }
    //~^ ERROR pointers cannot be reliably compared
}

fn main() {}
