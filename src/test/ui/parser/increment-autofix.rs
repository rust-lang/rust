// run-rustfix

fn post_regular() {
    let mut i = 0;
    i++; //~ ERROR Rust has no postfix increment operator
    println!("{}", i);
}

fn post_while() {
    let mut i = 0;
    while i++ < 5 {
        //~^ ERROR Rust has no postfix increment operator
        println!("{}", i);
    }
}

fn pre_regular() {
    let mut i = 0;
    ++i; //~ ERROR Rust has no prefix increment operator
    println!("{}", i);
}

fn pre_while() {
    let mut i = 0;
    while ++i < 5 {
        //~^ ERROR Rust has no prefix increment operator
        println!("{}", i);
    }
}

fn main() {
    post_regular();
    post_while();
    pre_regular();
    pre_while();
}
