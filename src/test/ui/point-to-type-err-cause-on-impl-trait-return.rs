fn foo() -> impl std::fmt::Display {
    if false {
        return 0i32;
    }
    1u32
    //~^ ERROR mismatched types
}

fn bar() -> impl std::fmt::Display {
    if false {
        return 0i32;
    } else {
        return 1u32;
        //~^ ERROR mismatched types
    }
}

fn baz() -> impl std::fmt::Display {
    if false {
    //~^ ERROR mismatched types
        return 0i32;
    } else {
        1u32
    }
}

fn qux() -> impl std::fmt::Display {
    if false {
        0i32
    } else {
        1u32
        //~^ ERROR if and else have incompatible types
    }
}

fn main() {}
