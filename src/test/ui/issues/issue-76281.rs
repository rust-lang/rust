// compile-flags: -C opt-level=2

fn main() {
    let mut v: Vec<&()> = Vec::new();
    v.sort_by_key(|&r| r as *const ());
}
