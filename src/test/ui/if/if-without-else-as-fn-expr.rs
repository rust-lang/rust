fn foo(bar: usize) -> usize {
    if bar % 5 == 0 {
        return 3;
    }
    //~^^^ ERROR if may be missing an else clause
}

fn main() {
    let _ = foo(1);
}
