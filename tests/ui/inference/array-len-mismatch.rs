fn returns_arr() -> [u8; 2] {
    [1, 2]
}

fn main() {
    let wrong: [u8; 3] = [10, 20];
    //~^ ERROR mismatched types
    //~^^ HELP consider specifying the actual array length
    let wrong: [u8; 3] = returns_arr();
    //~^ ERROR mismatched types
    //~^^ HELP consider specifying the actual array length
}
