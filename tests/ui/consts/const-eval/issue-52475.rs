fn main() {
    let _ = [(); {
        let mut x = &0;
        let mut n = 0;
        while n < 5 { //~ ERROR evaluation of constant value failed [E0080]
            n = (n + 1) % 5;
            x = &0; // Materialize a new AllocId
        }
        0
    }];
}
