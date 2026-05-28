fn test0() {
    let mut i = 0;
    let _ = i + i--; //~ ERROR Rust has no postfix decrement operator
    // won't suggest since we can not handle the precedences
}

fn test1() {
    let mut i = 0;
    let _ = i-- + i--; //~ ERROR Rust has no postfix decrement operator
}

fn test2() {
    let mut i = 0;
    let _ = --i + i--; //~ ERROR Rust has no postfix decrement operator
}

fn test3() {
    let mut i = 0;
    let _ = i-- + --i; //~ ERROR Rust has no postfix decrement operator
}

fn test4() {
    let mut i = 0;
    let _ = (1 + 2 + i)--; //~ ERROR Rust has no postfix decrement operator
}

fn test5() {
    let mut i = 0;
    let _ = (i-- + 1) + 2; //~ ERROR Rust has no postfix decrement operator
}

fn test6(){
    let i=10;
    while i != 0 {
        i--; //~ ERROR Rust has no postfix decrement operator
    }
}

fn main() {}
