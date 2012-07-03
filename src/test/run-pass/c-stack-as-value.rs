#[abi = "cdecl"]
extern mod rustrt {
    fn unsupervise();
}

fn main() {
    let _foo = rustrt::unsupervise;
}
