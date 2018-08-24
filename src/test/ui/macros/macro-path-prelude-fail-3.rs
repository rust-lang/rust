#[derive(inline)] //~ ERROR cannot find derive macro `inline` in this scope
struct S;

fn main() {
    inline!(); //~ ERROR cannot find macro `inline!` in this scope
}
