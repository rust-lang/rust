fn main() {
    const N: u32 = 1_000;
    const M: usize = (f64::from(N) * std::f64::LOG10_2) as usize; //~ ERROR cannot find value
    let mut digits = [0u32; M];
    //~^ ERROR evaluation of constant value failed
}
