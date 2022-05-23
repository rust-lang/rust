fn get_slice() -> &'static [i32] {
    &[1, 2, 3, 4]
}

fn main() {
    let sqsum = get_slice().map(|i| i * i).sum(); //~ ERROR [E0599]
}
