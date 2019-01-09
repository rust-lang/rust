// Test that an unboxed closure that captures a free variable by
// reference cannot escape the region of that variable.


fn main() {
    let _f = {
        let x = 0;
        || x //~ ERROR `x` does not live long enough
    };
    _f;
}
