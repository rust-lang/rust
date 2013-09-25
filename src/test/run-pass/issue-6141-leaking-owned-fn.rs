fn run(f: &fn()) {
    f()
}

pub fn main() {
    let f: ~fn() = || ();
    run(f);
}
