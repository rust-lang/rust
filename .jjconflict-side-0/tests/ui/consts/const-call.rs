fn f(x: usize) -> usize {
    x
}

fn main() {
    let _ = [0; f(2)];
    //~^ ERROR cannot call non-const function
}
