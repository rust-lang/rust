#![warn(clippy::all)]
#![warn(clippy::else_if_without_else)]

fn bla1() -> bool {
    unimplemented!()
}
fn bla2() -> bool {
    unimplemented!()
}
fn bla3() -> bool {
    unimplemented!()
}

fn main() {
    if bla1() {
        println!("if");
    }

    if bla1() {
        println!("if");
    } else {
        println!("else");
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if");
    } else {
        println!("else")
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if 1");
    } else if bla3() {
        println!("else if 2");
    } else {
        println!("else")
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        //~^ ERROR: `if` expression with an `else if`, but without a final `else`
        println!("else if");
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if 1");
    } else if bla3() {
        //~^ ERROR: `if` expression with an `else if`, but without a final `else`
        println!("else if 2");
    }
}
