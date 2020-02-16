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
        return 0i32;
    } else {
        1u32
        //~^ ERROR mismatched types
    }
}

fn qux() -> impl std::fmt::Display {
    if false {
        0i32
    } else {
        1u32
        //~^ ERROR `if` and `else` have incompatible types
    }
}

fn bat() -> impl std::fmt::Display {
    match 13 {
        0 => return 0i32,
        _ => 1u32,
        //~^ ERROR mismatched types
    }
}

fn can() -> impl std::fmt::Display {
    match 13 {
    //~^ ERROR mismatched types
        0 => return 0i32,
        1 => 1u32,
        _ => 2u32,
    }
}

fn cat() -> impl std::fmt::Display {
    match 13 {
        0 => {
            return 0i32;
        }
        _ => {
            1u32
            //~^ ERROR mismatched types
        }
    }
}

fn main() {}
