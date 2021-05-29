#![crate_type = "bin"]

fn main() {
    middle::do_a_thing();

    println!("middle::BAR {}", middle::BAR);

    assert_eq!(middle::simple(2, 3), 5);
}
