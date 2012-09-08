mod spam {
    fn ham() { }
    fn eggs() { }
}

#[abi = "cdecl"]
extern mod rustrt {
    use spam::{ham, eggs};
    export ham;
    export eggs;
}

fn main() { rustrt::ham(); rustrt::eggs(); }
