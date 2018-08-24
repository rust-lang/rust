// compile-flags: -Z parse-only

#[path = "not_a_real_file.rs"]
mod m; //~ ERROR not_a_real_file.rs

fn main() {
    assert_eq!(m::foo(), 10);
}
