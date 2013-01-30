fn main() {
    let mut a = [1, 2, 3, 4];
    let _ = match a {
        [1, 2, ..move tail] => tail,
        _ => core::util::unreachable()
    };
    a[0] = 0; //~ ERROR: use of moved value
}
