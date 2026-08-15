fn main() {
    if true {
    } else if {
    //~^ ERROR missing condition for `if` expression
    } else {
    }
}

fn foo() {
    if true {
    } else if {
    //~^ ERROR missing condition for `if` expression
    }
    bar();
}

fn bar() {}
