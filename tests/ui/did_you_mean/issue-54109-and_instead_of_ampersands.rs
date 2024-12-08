fn main() {}

fn test_and() {
    let a = true;
    let b = false;

    let _ = a and b; //~ ERROR `and` is not a logical operator

    if a and b { //~ ERROR `and` is not a logical operator
        println!("both");
    }

    let _recovery_witness: () = 0; //~ ERROR mismatched types
}

fn test_or() {
    let a = true;
    let b = false;

    let _ = a or b; //~ ERROR `or` is not a logical operator

    if a or b { //~ ERROR `or` is not a logical operator
        println!("both");
    }
}

fn test_and_par() {
    let a = true;
    let b = false;
    if (a and b) {  //~ ERROR `and` is not a logical operator
        println!("both");
    }
}

fn test_or_par() {
    let a = true;
    let b = false;
    if (a or b) {  //~ ERROR `or` is not a logical operator
        println!("both");
    }
}

fn test_while_and() {
    let a = true;
    let b = false;
    while a and b {  //~ ERROR `and` is not a logical operator
        println!("both");
    }
}

fn test_while_or() {
    let a = true;
    let b = false;
    while a or b { //~ ERROR `or` is not a logical operator
        println!("both");
    }
}
