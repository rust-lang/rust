extern crate dep_one;
extern crate dep_two;

fn main() {
    println!("{}", dep_one::foo() + dep_two::bar());
}
