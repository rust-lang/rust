// run-rustfix

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

fn pre_regular_tmp() {
    let mut tmp = 0;
    ++tmp; //~ ERROR Rust has no prefix increment operator
    println!("{}", tmp);
}

fn pre_while_tmp() {
    let mut tmp = 0;
    while ++tmp < 5 {
        //~^ ERROR Rust has no prefix increment operator
        println!("{}", tmp);
    }
}

fn main() {}
