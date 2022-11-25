fn test1() {
    let mut i = 0;
    let _ = i + ++i; //~ ERROR Rust has no prefix increment operator
}

fn test2() {
    let mut i = 0;
    let _ = ++i + i; //~ ERROR Rust has no prefix increment operator
}

fn test3() {
    let mut i = 0;
    let _ = ++i + ++i; //~ ERROR Rust has no prefix increment operator
}

fn test4() {
    let mut i = 0;
    let _ = i + i++; //~ ERROR Rust has no postfix increment operator
}

fn test5() {
    let mut i = 0;
    let _ = i++ + i; //~ ERROR Rust has no postfix increment operator
}

fn test6() {
    let mut i = 0;
    let _ = i++ + i++; //~ ERROR Rust has no postfix increment operator
}

fn test7() {
    let mut i = 0;
    let _ = ++i + i++; //~ ERROR Rust has no prefix increment operator
}

fn test8() {
    let mut i = 0;
    let _ = i++ + ++i; //~ ERROR Rust has no postfix increment operator
}

fn main() { }
