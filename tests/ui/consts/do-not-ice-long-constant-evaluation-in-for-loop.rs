// Issue-125323
fn main() {
    for _ in 0..0 { //~ ERROR type annotations needed [E0282]
        [(); loop {}]; //~ ERROR constant evaluation is taking a long time
    }
}
