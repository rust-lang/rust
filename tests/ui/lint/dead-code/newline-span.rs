#![deny(dead_code)]

fn unused() { //~ error: function `unused` is never used
    println!("blah");
}

fn unused2(var: i32) { //~ error: function `unused2` is never used
    println!("foo {}", var);
}

fn unused3( //~ error: function `unused3` is never used
    var: i32,
) {
    println!("bar {}", var);
}

fn main() {
    println!("Hello world!");
}
