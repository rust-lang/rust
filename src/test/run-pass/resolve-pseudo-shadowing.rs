// run-pass
// check that type parameters can't "shadow" qualified paths.

fn check<Clone>(_c: Clone) {
    fn check2() {
        let _ = <() as std::clone::Clone>::clone(&());
    }
    check2();
}

fn main() { check(()); }
