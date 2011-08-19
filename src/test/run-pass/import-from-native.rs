mod spam {
    fn ham() { }
    fn eggs() { }
}

native "rust" mod rustrt {
    import spam::{ham, eggs};
    export ham;
    export eggs;
}

fn main() { rustrt::ham(); rustrt::eggs(); }
