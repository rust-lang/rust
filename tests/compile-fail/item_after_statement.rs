#![feature(plugin)]
#![plugin(clippy)]
#![deny(items_after_statements)]

fn main() {
    foo();
    fn foo() { println!("foo"); } //~ ERROR adding items after statements is confusing
    foo();
}
