// ABI is cdecl by default

extern mod rustrt {
    fn unsupervise();
}

fn main() {
    rustrt::unsupervise();
}