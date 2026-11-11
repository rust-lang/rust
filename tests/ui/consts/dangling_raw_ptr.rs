const FOO: *const u32 = {
    //~^ ERROR encountered dangling pointer in final value of constant
    let x = 42;
    &x
};

union Union {
    ptr: *const u32,
}

const BAR: Union = {
    //~^ ERROR encountered dangling pointer in final value of constant
    let x = 42;
    Union { ptr: &x }
};

const BAZ: Union = {
    //~^ ERROR encountered dangling pointer in final value of constant
    let x = 42_u32;
    Union { ptr: &(&x as *const u32) as *const *const u32 as _ }
};

const FOOMP: *const u32 = {
    //~^ ERROR encountered dangling pointer in final value of constant
    let x = 42_u32;
    &(&x as *const u32) as *const *const u32 as _
};

fn main() {}
