fn test4() {
    let mut i = 0;
    let _ = i + i--; //~ ERROR Rust has no postfix decrement operator
    // won't suggest since we can not handle the precedences
}

fn test5() {
    let mut i = 0;
    let _ = i-- + i--; //~ ERROR Rust has no postfix decrement operator
}

fn test6() {
    let mut i = 0;
    let _ = --i + i--; //~ ERROR Rust has no postfix decrement operator
}

fn test7() {
    let mut i = 0;
    let _ = i-- + --i; //~ ERROR Rust has no postfix decrement operator
}

fn test8() {
    let mut i = 0;
    let _ = (1 + 2 + i)--; //~ ERROR Rust has no postfix decrement operator
}

fn test9() {
    let mut i = 0;
    let _ = (i-- + 1) + 2; //~ ERROR Rust has no postfix decrement operator
}



fn test14(){
    let i=10;
    while i!=0{
        i--; //~ ERROR Rust has no postfix decrement operator
    }
}

fn main() { }
