const fn cmp(x: fn(), y: fn()) -> bool { //~ ERROR function pointers in const fn are unstable
    unsafe { x == y }
}

fn main() {}
