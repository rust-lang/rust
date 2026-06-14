const X : usize = 2;

const fn f(x: usize) -> usize {
    let mut sum = 0;
    for i in 0..x {
        //~^ ERROR cannot use `for`
        //~| ERROR `IntoIterator` is not yet stable
        //~| ERROR cannot use `for`
        //~| ERROR `Iterator` is not yet stable
        sum += i;
    }
    sum
}

#[allow(unused_variables)]
fn main() {
    let a : [i32; f(X)];
}
