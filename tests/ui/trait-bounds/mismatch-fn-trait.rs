fn take(_f: impl FnMut(i32)) {}

fn test1(f: impl FnMut(u32)) {
    take(f)
    //~^ ERROR [E0277]
}

fn test2(f: impl FnMut(i32, i32)) {
    take(f)
    //~^ ERROR [E0277]
}

fn test3(f: impl FnMut()) {
    take(f)
    //~^ ERROR [E0277]
}

fn test4(f: impl FnOnce(i32)) {
    take(f)
    //~^ ERROR [E0277]
}

fn test5(f: impl FnOnce(u32)) {
    take(f)
    //~^ ERROR [E0277]
}

fn main() {}
