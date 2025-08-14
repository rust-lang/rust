//@ edition: 2024
//@ run-rustfix
#![feature(gen_blocks)]

fn moved() -> impl Iterator<Item = u32> {
    let mut x = "foo".to_string();
    gen { //~ ERROR: gen block may outlive the current function
        yield 42;
        if x == "foo" { return }
        x.clear();
        for x in 3..6 { yield x }
    }
}

fn main() {
    for _ in moved() {}
}
