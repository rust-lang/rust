fn main() {
    let mut a = [1, 2, 3, 4];
    let _ = match a {
        [1, 2, ..tail] => tail,
        _ => core::util::unreachable()
    };
    a[0] = 0; //~ ERROR: assigning to mutable vec content prohibited due to outstanding loan
}
