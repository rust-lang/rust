#![warn(clippy::else_if_without_else)]
#![allow(clippy::collapsible_else_if)]

fn bla1() -> bool {
    unimplemented!()
}
fn bla2() -> bool {
    unimplemented!()
}
fn bla3() -> bool {
    unimplemented!()
}
fn bla4() -> bool {
    unimplemented!()
}
fn bla5() -> bool {
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
        //~^ else_if_without_else

        println!("else if");
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if 1");
    } else if bla3() {
        //~^ else_if_without_else

        println!("else if 2");
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if 1");
    } else if bla3() {
        println!("else if 2");
    } else if bla4() {
        println!("else if 3");
    } else if bla5() {
        println!("else if 4");
    } else {
        println!("else");
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if 1");
    } else if bla3() {
        println!("else if 2");
    } else if bla4() {
        println!("else if 3");
    } else if bla5() {
        //~^ else_if_without_else

        println!("else if 4");
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if 1");
    } else {
        if bla3() {
            println!("else if 2");
        } else if bla4() {
            println!("else if 3");
        } else if bla5() {
            println!("else if 4");
        } else {
            println!("else");
        }
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if 1");
    } else {
        if bla3() {
            println!("else if 2");
        } else if bla4() {
            println!("else if 3");
        } else if bla5() {
            //~^ else_if_without_else

            println!("else if 4");
        }
    }
}
