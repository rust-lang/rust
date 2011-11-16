// ABI is cdecl by default

native mod rustrt {
    fn unsupervise();
}

fn main() {
    rustrt::unsupervise();
}