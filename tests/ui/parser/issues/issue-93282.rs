fn main() {
    f<'a,>
    //~^ ERROR expected
    //~| ERROR expected
}

fn bar(a: usize, b: usize) -> usize {
    a + b
}

fn foo() {
    let x = 1;
    bar('y, x);
    //~^ ERROR expected
    //~| ERROR mismatched types
}
