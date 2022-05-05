const FOO: *const u32 = {
    let x;
    &x //~ ERROR borrow of possibly-uninitialized variable: `x`
};

fn main() {
    let FOO = FOO;
    //~^ ERROR could not evaluate constant pattern
    //~| ERROR could not evaluate constant pattern
}
