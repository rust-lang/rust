fn foo() -> impl std::fmt::Display {
    if false {
        return 0i32;
    }
    1u32 //~ ERROR mismatched types
}

fn bar() -> impl std::fmt::Display {
    if false {
        return 0i32;
    } else {
        return 1u32; //~ ERROR mismatched types
    }
}

fn baz() -> impl std::fmt::Display {
    if false {
        return 0i32;
    } else {
        1u32 //~ ERROR mismatched types
    }
}

fn qux() -> impl std::fmt::Display {
    if false {
        0i32
    } else {
        1u32 //~ ERROR `if` and `else` have incompatible types
    }
}

fn bat() -> impl std::fmt::Display {
    match 13 {
        0 => return 0i32,
        _ => 1u32, //~ ERROR mismatched types
    }
}

fn can() -> impl std::fmt::Display {
    match 13 { //~ ERROR mismatched types
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
            1u32 //~ ERROR mismatched types
        }
    }
}

fn dog() -> impl std::fmt::Display {
    match 13 {
        0 => 0i32,
        1 => 1u32, //~ ERROR `match` arms have incompatible types
        _ => 2u32,
    }
}

fn hat() -> dyn std::fmt::Display { //~ ERROR return type cannot be a trait object without pointer indirection
    match 13 {
        0 => {
            return 0i32;
        }
        _ => {
            1u32
        }
    }
}

fn pug() -> dyn std::fmt::Display { //~ ERROR return type cannot be a trait object without pointer indirection
    match 13 {
        0 => 0i32,
        1 => 1u32,
        _ => 2u32,
    }
}

fn man() -> dyn std::fmt::Display { //~ ERROR return type cannot be a trait object without pointer indirection
    if false {
        0i32
    } else {
        1u32
    }
}

fn apt() -> impl std::fmt::Display {
    if let Some(42) = Some(42) {
        0i32
    } else {
        1u32 //~ ERROR `if` and `else` have incompatible types
    }
}

fn main() {}
