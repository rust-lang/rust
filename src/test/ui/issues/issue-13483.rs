fn main() {
    if true {
    } else if { //~ ERROR missing condition
    } else {
    }
}

fn foo() {
    if true {
    } else if { //~ ERROR missing condition
    }
    bar();
}

fn bar() {}
