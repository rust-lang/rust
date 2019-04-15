fn main() {}

const fn slice([a, b]: &[i32]) -> i32 { //~ ERROR refutable pattern in function argument
    a + b //~ ERROR can only call other `const fn` within a `const fn`
    //~^ WARN use of possibly uninitialized variable: `a`
    //~| WARN this error has been downgraded to a warning for backwards compatibility
    //~| WARN this represents potential undefined behavior in your code and this warning will
    //~| WARN use of possibly uninitialized variable: `b`
    //~| WARN this error has been downgraded to a warning for backwards compatibility
    //~| WARN this represents potential undefined behavior in your code and this warning will
}
