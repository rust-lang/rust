fn main() {
    let v = vec![0];
    const l: usize = v.count(); //~ ERROR can't capture dynamic environment in a fn item
    let s: [u32; l] = v.into_iter().collect();
    //~^ ERROR evaluation of constant value failed
}
