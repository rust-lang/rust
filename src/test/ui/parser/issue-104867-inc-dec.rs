struct S {
    x: i32,
}

fn test1() {
    let mut i = 0;
    i++; //~ ERROR Rust has no postfix increment operator
}

fn test2() {
    let s = S { x: 0 };
    s.x++; //~ ERROR Rust has no postfix increment operator
}

fn test3() {
    let mut i = 0;
    if i++ == 1 {} //~ ERROR Rust has no postfix increment operator
}

fn test4() {
    let mut i = 0;
    ++i; //~ ERROR Rust has no prefix increment operator
}

fn test5() {
    let mut i = 0;
    if ++i == 1 { } //~ ERROR Rust has no prefix increment operator
}

fn test6() {
    let mut i = 0;
    loop { break; }
    i++; //~ ERROR Rust has no postfix increment operator
    loop { break; }
    ++i;
}

fn test7() {
    let mut i = 0;
    loop { break; }
    ++i; //~ ERROR Rust has no prefix increment operator
}


fn main() {}
