fn test_and() {
    let a = true;
    let b = false;
    if a and b {
        //~^ ERROR expected `{`, found `and`
        println!("both");
    }
}

fn test_or() {
    let a = true;
    let b = false;
    if a or b {
        //~^ ERROR expected `{`, found `or`
        println!("both");
    }
}

fn test_and_par() {
    let a = true;
    let b = false;
    if (a and b) {
        //~^ ERROR expected one of `!`, `)`, `,`, `.`, `::`, `?`, `{`, or an operator, found `and`
        println!("both");
    }
}

fn test_or_par() {
    let a = true;
    let b = false;
    if (a or b) {
        //~^ ERROR expected one of `!`, `)`, `,`, `.`, `::`, `?`, `{`, or an operator, found `or`
        println!("both");
    }
}

fn test_while_and() {
    let a = true;
    let b = false;
    while a and b {
        //~^ ERROR expected one of `!`, `.`, `::`, `?`, `{`, or an operator, found `and`
        println!("both");
    }
}

fn test_while_or() {
    let a = true;
    let b = false;
    while a or b {
        //~^ ERROR expected one of `!`, `.`, `::`, `?`, `{`, or an operator, found `or`
        println!("both");
    }
}

fn main() {
}
