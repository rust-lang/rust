//@ ignore-auxiliary (used by `./move-error-snippets.rs`)

macro_rules! aaa {
    ($c:ident) => {{
        let a = $c;
    }}
}
