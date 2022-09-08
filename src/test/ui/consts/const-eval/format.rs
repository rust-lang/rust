const fn failure() {
    panic!("{:?}", 0);
    //~^ ERROR
    //~| ERROR
    //~| ERROR
    //~| ERROR
}

const fn print() {
    println!("{:?}", 0);
    //~^ ERROR
    //~| ERROR
    //~| ERROR
    //~| WARN
}

fn main() {}
