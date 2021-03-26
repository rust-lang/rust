fn test1() {
    let i = 0;
    i++; //~ ERROR
}

fn test2() {
    let i = 0;
    ++i; //~ ERROR
}

fn main() {}
