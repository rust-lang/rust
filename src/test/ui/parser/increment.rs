fn post_regular() {
    let i = 0;
    i++; //~ ERROR
}

fn post_while() {
    let i = 0;
    while i++ < 5 {
        //~^ ERROR
        println!("{}", i);
    }
}

fn pre_regular() {
    let i = 0;
    ++i; //~ ERROR
}

fn pre_while() {
    let i = 0;
    while ++i < 5 {
        //~^ ERROR
        println!("{}", i);
    }
}

fn main() {}
