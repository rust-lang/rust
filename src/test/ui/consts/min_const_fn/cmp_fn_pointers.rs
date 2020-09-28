const fn cmp(x: fn(), y: fn()) -> bool { //~ ERROR function pointer
    unsafe { x == y }
}

fn main() {}
