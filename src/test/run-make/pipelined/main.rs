#![crate_type = "bin"]

fn main() {
    middle::do_a_thing();

    println!("middle::BAR {}", middle::BAR);

    assert_eq!(middle::simple(2, 3), 5);
    assert_eq!(middle::inlined(2, 3), 6);
}
