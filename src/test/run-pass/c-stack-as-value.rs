// xfail-test

native "cdecl" mod rustrt {
    fn unsupervise();
}

fn main() {
    let _foo = rustrt::unsupervise;
}
