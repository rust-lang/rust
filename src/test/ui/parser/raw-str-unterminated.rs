// compile-flags: -Z parse-only

static s: &'static str =
    r#" string literal goes on
        and on
    //~^^ ERROR unterminated raw string
