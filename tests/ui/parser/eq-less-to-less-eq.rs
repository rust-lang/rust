fn foo() {
    let a = 0;
    let b = 4;
    if a =< b { //~ERROR
        println!("yay!");
    }
}

fn bar() {
    let a = 0;
    let b = 4;
    if a = <b { //~ERROR
        println!("yay!");
    }
}

fn baz() {
    let a = 0;
    let b = 4;
    if a = < b { //~ERROR
        println!("yay!");
    }
}

fn qux() {
    let a = 0;
    let b = 4;
    if a =< i32>::abs(-4) { //~ERROR: mismatched types
        println!("yay!");
    }
}

fn main() {}
