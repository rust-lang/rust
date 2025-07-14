use std::ops::Add;

fn dbl<T>(x: T) -> <T as Add>::Output
where
    T: Copy + Add,
    UUU: Copy,
    //~^ ERROR cannot find type `UUU` in this scope
{
    x + x
    //~^ ERROR use of moved value: `x`
}

fn main() {
    println!("{}", dbl(3));
}
