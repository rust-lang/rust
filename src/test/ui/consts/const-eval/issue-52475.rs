fn main() {
    let _ = [(); {
        let mut x = &0;
        let mut n = 0;
        while n < 5 {
            n = (n + 1) % 5; //~ ERROR evaluation of constant value failed
            x = &0; // Materialize a new AllocId
        }
        0
    }];
}
