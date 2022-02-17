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

fn post_regular_tmp() {
    let mut tmp = 0;
    tmp++; //~ ERROR Rust has no postfix increment operator
    println!("{}", tmp);
}

fn post_while_tmp() {
    let mut tmp = 0;
    while tmp++ < 5 {
        //~^ ERROR Rust has no postfix increment operator
        println!("{}", tmp);
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
    post_regular_tmp();
    post_while_tmp();
    pre_regular();
    pre_while();
}
