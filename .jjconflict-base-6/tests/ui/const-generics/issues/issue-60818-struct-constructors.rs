//@ check-pass

struct Generic<const V: usize>;

fn main() {
    let _ = Generic::<0>;
}
