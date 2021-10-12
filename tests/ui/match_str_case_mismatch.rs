#![warn(clippy::match_str_case_mismatch)]

// Valid

fn as_str_match() {
    let var = "BAR";

    match var.to_ascii_lowercase().as_str() {
        "foo" => {},
        "bar" => {},
        _ => {},
    }
}

fn addrof_unary_match() {
    let var = "BAR";

    match &*var.to_ascii_lowercase() {
        "foo" => {},
        "bar" => {},
        _ => {},
    }
}

fn alternating_chain() {
    let var = "BAR";

    match &*var
        .to_ascii_lowercase()
        .to_uppercase()
        .to_lowercase()
        .to_ascii_uppercase()
    {
        "FOO" => {},
        "BAR" => {},
        _ => {},
    }
}

fn unrelated_method() {
    struct Item {
        a: String,
    }

    impl Item {
        #[allow(clippy::wrong_self_convention)]
        fn to_lowercase(self) -> String {
            self.a
        }
    }

    let item = Item { a: String::from("BAR") };

    match &*item.to_lowercase() {
        "FOO" => {},
        "BAR" => {},
        _ => {},
    }
}

// Invalid

fn as_str_match_mismatch() {
    let var = "BAR";

    match var.to_ascii_lowercase().as_str() {
        "foo" => {},
        "Bar" => {},
        _ => {},
    }
}

fn addrof_unary_match_mismatch() {
    let var = "BAR";

    match &*var.to_ascii_lowercase() {
        "foo" => {},
        "Bar" => {},
        _ => {},
    }
}

fn alternating_chain_mismatch() {
    let var = "BAR";

    match &*var
        .to_ascii_lowercase()
        .to_uppercase()
        .to_lowercase()
        .to_ascii_uppercase()
    {
        "FOO" => {},
        "bAR" => {},
        _ => {},
    }
}

fn main() {}
