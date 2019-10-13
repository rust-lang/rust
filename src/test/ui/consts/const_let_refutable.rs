fn main() {}

const fn slice([a, b]: &[i32]) -> i32 { //~ ERROR refutable pattern in function argument
    a + b //~ ERROR can only call other `const fn` within a `const fn`
    //~^ ERROR use of possibly-uninitialized variable: `a`
    //~| ERROR use of possibly-uninitialized variable: `b`
}
