struct MyStruct;

trait Test {
    const TEST: fn() -> _;
    //~^ ERROR: the type placeholder `_` is not allowed within types on item signatures for functions [E0121]
    //~| ERROR: the type placeholder `_` is not allowed within types on item signatures for constants [E0121]
}

impl Test for MyStruct {
    const TEST: fn() -> _ = 42;
    //~^ ERROR: the type placeholder `_` is not allowed within types on item signatures for functions [E0121]
}

fn main() {}
