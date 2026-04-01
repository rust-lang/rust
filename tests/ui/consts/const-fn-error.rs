const X : usize = 2;

const fn f(x: usize) -> usize {
    let mut sum = 0;
    for i in 0..x {
        //~^ ERROR `std::ops::Range<usize>: [const] Iterator` is not satisfied
        //~| ERROR `std::ops::Range<usize>: [const] Iterator` is not satisfied
        sum += i;
    }
    sum
}

#[allow(unused_variables)]
fn main() {
    let a : [i32; f(X)];
}
