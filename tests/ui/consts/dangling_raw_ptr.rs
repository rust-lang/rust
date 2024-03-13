const FOO: *const u32 = { //~ ERROR it is undefined behavior
    let x = 42;
    &x
};

union Union {
    ptr: *const u32
}

const BAR: Union = { //~ ERROR encountered dangling pointer in final value
    let x = 42;
    Union { ptr: &x }
};

fn main() {
    let x = FOO;
    let x = BAR;
}
