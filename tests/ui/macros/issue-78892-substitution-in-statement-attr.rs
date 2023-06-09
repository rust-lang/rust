// check-pass

// regression test for #78892

macro_rules! mac {
    ($lint_name:ident) => {{
        #[allow($lint_name)]
        let _ = ();
    }};
}

fn main() {
    mac!(dead_code)
}
