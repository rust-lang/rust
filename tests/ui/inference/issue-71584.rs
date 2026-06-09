//@ ignore-windows FIXME: We get an extra E0283 on Windows
fn main() {
    let n: u32 = 1;
    let mut d: u64 = 2;
    d = d % n.into();
    //~^ ERROR type annotations needed
}
