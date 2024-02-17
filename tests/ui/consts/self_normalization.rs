//@ check-pass

fn testfn(_arr: &mut [(); 0]) {}

trait TestTrait {
    fn method();
}

impl TestTrait for [(); 0] {
    fn method() {
        let mut arr: Self = [(); 0];
        testfn(&mut arr);
    }
}

fn main() {}
