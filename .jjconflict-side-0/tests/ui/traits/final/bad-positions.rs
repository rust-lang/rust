#![feature(final_associated_functions)]

fn main() {
    final; //~ ERROR `final` is not followed by an item
    final 1 + 1; //~ ERROR `final` is not followed by an item
    final { println!("text"); }; //~ ERROR `final` is not followed by an item
}
