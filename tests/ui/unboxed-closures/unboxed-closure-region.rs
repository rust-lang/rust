// Test that an unboxed closure that captures a free variable by
// reference cannot escape the region of that variable.


fn main() {
    let _f = {
        let x = 0;
        || x //~ ERROR closure may outlive the current block, but it borrows `x`
    };
    _f;
}
