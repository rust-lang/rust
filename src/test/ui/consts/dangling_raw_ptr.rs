const FOO: *const u32 = { //~ ERROR any use of this value will cause an error
    let x = 42;
    &x
};

fn main() {
    let x = FOO;
}
