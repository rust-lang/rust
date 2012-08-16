class c { //~ ERROR a struct must have at least one field
    new() { }
}

fn main() {
    let a = c();
    let x = ~[a];
    let _y = x[0];
}
