fn main() {}

const fn slice(&[a, b]: &[i32]) -> i32 {
    //~^ ERROR refutable pattern in function argument
    //~| ERROR loops and conditional expressions are not stable in const fn
    a + b
}
