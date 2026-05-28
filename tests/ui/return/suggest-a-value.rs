fn test() -> (i32,) {
    return;
    //~^ ERROR `return;` in a function whose return type is not `()`
}

fn main() {}
