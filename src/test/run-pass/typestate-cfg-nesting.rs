

fn f() { let x = 10; let y = 11; if true { alt x { _ { y = x; } } } else { } }

fn main() {
    let x = 10;
    let y = 11;
    if true { while false { y = x; } } else { }
}