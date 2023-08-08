const FOO: *const u32 = {
    let x;
    &x //~ ERROR E0381
};

fn main() {
    let FOO = FOO;
    //~^ ERROR could not evaluate constant pattern
}
