const X : usize = 2;

const fn f(x: usize) -> usize {
    let mut sum = 0;
    for i in 0..x {
        //~^ ERROR mutable references
        //~| ERROR calls in constant functions
        //~| ERROR calls in constant functions
        //~| ERROR E0080
        //~| ERROR `for` is not allowed in a `const fn`
        sum += i;
    }
    sum
}

#[allow(unused_variables)]
fn main() {
    let a : [i32; f(X)];
}
