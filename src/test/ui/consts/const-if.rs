const _: i32 = if true { //~ ERROR `if` is not allowed in a `const`
    5
} else {
    6
};

const _: i32 = match 1 { //~ ERROR `match` is not allowed in a `const`
    2 => 3,
    4 => 5,
    _ => 0,
};

const fn foo() -> i32 {
    if true { 5 } else { 6 } //~ ERROR `if` is not allowed in a `const fn`
}

const fn bar() -> i32 {
    match 0 { 1 => 2, _ => 0 } //~ ERROR `match` is not allowed in a `const fn`
}

fn main() {}
