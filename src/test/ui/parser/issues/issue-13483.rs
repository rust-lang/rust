fn main() {
    if true {
    } else if { //~ ERROR missing condition
    //~^ ERROR mismatched types
    } else {
    }
}

fn foo() {
    if true {
    } else if { //~ ERROR missing condition
    //~^ ERROR mismatched types
    }
    bar();
}

fn bar() {}
