const FOO: *const u32 = { //~ ERROR it is undefined behavior
    let x = 42;
    &x
};

fn main() {
    let x = FOO;
}
