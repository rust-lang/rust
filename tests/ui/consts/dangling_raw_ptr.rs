const FOO: *const u32 = { //~ ERROR encountered dangling pointer in final value of constant
    let x = 42;
    &x
};

fn main() {
    let x = FOO;
}
