const FOO: *const u32 = { //~ ERROR encountered dangling pointer in final constant
    let x;
    &x //~ ERROR borrow of possibly-uninitialized variable: `x`
};

fn main() {
    let FOO = FOO;
    //~^ ERROR could not evaluate constant pattern
    //~| ERROR could not evaluate constant pattern
}
