#[path = "not_a_real_file.rs"]
mod m; //~ ERROR not_a_real_file.rs

fn main() {
    assert m::foo() == 10;
}