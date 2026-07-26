macro_rules! var_named_via_macro {
    () => {
        "NON_UNICODE_VAR"
    };
}

fn main() {
    let _ = env!("NON_UNICODE_VAR");
    let _ = option_env!("NON_UNICODE_VAR");
    let _ = env!(var_named_via_macro!());
    let _ = option_env!(var_named_via_macro!());
}
