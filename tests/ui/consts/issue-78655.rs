const FOO: *const u32 = {
    let x;
    &x //~ ERROR E0381
};

fn main() {
    let FOO = FOO; // ok, the `const` already emitted an error
}
