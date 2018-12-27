// Test use of const let without feature gate.

const FOO: usize = {
    //~^ ERROR statements in constants are unstable
    //~| ERROR: let bindings in constants are unstable
    let x = 42;
    //~^ ERROR statements in constants are unstable
    //~| ERROR: let bindings in constants are unstable
    42
};

static BAR: usize = {
    //~^ ERROR statements in statics are unstable
    //~| ERROR: let bindings in statics are unstable
    let x = 42;
    //~^ ERROR statements in statics are unstable
    //~| ERROR: let bindings in statics are unstable
    42
};

fn main() {}
