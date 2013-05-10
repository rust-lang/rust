fn run(f: &fn()) {
    f()
}

fn main() {
    let f: ~fn() = || ();
    run(f);
}