const X : usize = 2;

const fn f(x: usize) -> usize {
    let mut sum = 0;
    for i in 0..x {
        //~^ ERROR mutable references
        //~| ERROR cannot convert
        //~| ERROR cannot call non-const fn
        //~| ERROR `for` is not allowed in a `const fn`
        sum += i;
    }
    sum
}

#[allow(unused_variables)]
fn main() {
    let a : [i32; f(X)];
}
