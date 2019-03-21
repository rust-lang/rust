fn foo(bar: usize) -> usize {
    if bar % 5 == 0 {
        return 3;
    }
    //~^^^ ERROR if may be missing an else clause
}

fn foo2(bar: usize) -> usize {
    let x: usize = if bar % 5 == 0 {
        return 3;
    };
    //~^^^ ERROR if may be missing an else clause
    x
}

fn foo3(bar: usize) -> usize {
    if bar % 5 == 0 {
        3
    }
    //~^^^ ERROR if may be missing an else clause
}

fn main() {
    let _ = foo(1);
}
