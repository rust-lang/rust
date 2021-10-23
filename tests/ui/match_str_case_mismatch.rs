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

fn non_alphabetic() {
    let var = "~!@#$%^&*()-_=+FOO";

    match var.to_ascii_lowercase().as_str() {
        "1234567890" => {},
        "~!@#$%^&*()-_=+foo" => {},
        "\n\r\t\x7F" => {},
        _ => {},
    }
}

fn unicode_cased() {
    let var = "ВОДЫ";

    match var.to_lowercase().as_str() {
        "水" => {},
        "νερό" => {},
        "воды" => {},
        "물" => {},
        _ => {},
    }
}

fn titlecase() {
    let var = "Barǲ";

    match var.to_lowercase().as_str() {
        "fooǉ" => {},
        "barǳ" => {},
        _ => {},
    }
}

fn no_case_equivalent() {
    let var = "barʁ";

    match var.to_uppercase().as_str() {
        "FOOɕ" => {},
        "BARʁ" => {},
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

fn non_alphabetic_mismatch() {
    let var = "~!@#$%^&*()-_=+FOO";

    match var.to_ascii_lowercase().as_str() {
        "1234567890" => {},
        "~!@#$%^&*()-_=+Foo" => {},
        "\n\r\t\x7F" => {},
        _ => {},
    }
}

fn unicode_cased_mismatch() {
    let var = "ВОДЫ";

    match var.to_lowercase().as_str() {
        "水" => {},
        "νερό" => {},
        "Воды" => {},
        "물" => {},
        _ => {},
    }
}

fn titlecase_mismatch() {
    let var = "Barǲ";

    match var.to_lowercase().as_str() {
        "fooǉ" => {},
        "barǲ" => {},
        _ => {},
    }
}

fn no_case_equivalent_mismatch() {
    let var = "barʁ";

    match var.to_uppercase().as_str() {
        "FOOɕ" => {},
        "bARʁ" => {},
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
