fn main() {
    let (0 | (1 | 2)) = 0; //~ ERROR refutable pattern in local binding
    match 0 {
        //~^ ERROR non-exhaustive patterns
        0 | (1 | 2) => {}
    }
}
