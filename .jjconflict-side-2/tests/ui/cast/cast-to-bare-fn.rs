fn foo(_x: isize) { }

fn main() {
    let v: u64 = 5;
    let x = foo as extern "C" fn() -> isize;
    //~^ ERROR non-primitive cast
    let y = v as extern "Rust" fn(isize) -> (isize, isize);
    //~^ ERROR non-primitive cast
    y(x());
}
