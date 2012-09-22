mod spam {
    #[legacy_exports];
    fn ham() { }
    fn eggs() { }
}

#[abi = "cdecl"]
extern mod rustrt {
    #[legacy_exports];
    use spam::{ham, eggs};
    export ham;
    export eggs;
}

fn main() { rustrt::ham(); rustrt::eggs(); }
