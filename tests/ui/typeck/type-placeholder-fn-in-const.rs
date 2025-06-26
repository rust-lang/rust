struct MyStruct;

trait Test {
    const TEST: fn() -> _;
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for associated constants [E0121]
}

impl Test for MyStruct {
    const TEST: fn() -> _ = 42;
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for associated constants [E0121]
}

fn main() {}
