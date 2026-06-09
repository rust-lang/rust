fn main() {
    let v = vec![0];
    const l: usize = v.count(); //~ ERROR attempt to use a non-constant value in a constant
    let s: [u32; l] = v.into_iter().collect();
}
