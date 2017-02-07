#![feature(plugin)]
#![plugin(clippy)]
#![deny(items_after_statements)]

fn ok() {
    fn foo() { println!("foo"); }
    foo();
}

fn last() {
    foo();
    fn foo() { println!("foo"); } //~ ERROR adding items after statements is confusing
}

fn main() {
    foo();
    fn foo() { println!("foo"); } //~ ERROR adding items after statements is confusing
    foo();
}
