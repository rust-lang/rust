// xfail-test

native "c-stack-cdecl" mod rustrt {
    fn unsupervise();
}

fn main() {
    let _foo = rustrt::unsupervise;
}
