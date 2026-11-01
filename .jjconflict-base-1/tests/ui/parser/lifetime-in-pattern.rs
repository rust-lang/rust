fn test(&'a str) {
    //~^ ERROR unexpected lifetime `'a` in pattern
    //~| ERROR expected one of `:`, `@`, or `|`, found `)`
}

fn main() {
}
