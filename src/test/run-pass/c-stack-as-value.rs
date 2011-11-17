// xfail-test

#[abi = "cdecl"]
native mod rustrt {
    fn unsupervise();
}

fn main() {
    let _foo = rustrt::unsupervise;
}
