// check-fail

macro_rules! one_token   { ( $a:tt ) => { }; }
macro_rules! two_tokens { ( $a:tt $b:tt ) => { }; }
macro_rules! three_tokens { ( $a:tt $b:tt $c:tt) => { }; }
macro_rules! four_tokens { ( $a:tt $b:tt $c:tt $d:tt ) => { }; }

fn main() {
    // Intended raw keyword syntax
    one_token!(k#ident);
    //~^ ERROR raw keyword syntax is reserved for future use
    one_token!(k#fn);
    //~^ ERROR raw keyword syntax is reserved for future use
    one_token!(k#Self);
    //~^ ERROR raw keyword syntax is reserved for future use
    one_token!(k#dyn);
    //~^ ERROR raw keyword syntax is reserved for future use
    one_token!(k#await);
    //~^ ERROR raw keyword syntax is reserved for future use

    // This was previously valid in `quote!` macros, where `k` was a variable.
    two_tokens!(#k#other_variable);
    //~^ ERROR raw keyword syntax is reserved for future use

    // weird interactions with raw literals.
    two_tokens!(k#r"raw string");
    //~^ ERROR raw keyword syntax is reserved for future use
    two_tokens!(k#r "raw string");
    //~^ ERROR raw keyword syntax is reserved for future use
    two_tokens!(k#br"raw byte string");
    //~^ ERROR raw keyword syntax is reserved for future use
    three_tokens!(k#r"raw string"#);
    //~^ ERROR raw keyword syntax is reserved for future use
    four_tokens!(k#r#"raw string"#);
    //~^ ERROR raw keyword syntax is reserved for future use
    two_tokens!(k#b'-');
    //~^ ERROR raw keyword syntax is reserved for future use
    two_tokens!(k#b '-');
    //~^ ERROR raw keyword syntax is reserved for future use

    // weird raw idents
    one_token!(r#k);
    two_tokens!(r#k#);
    three_tokens!(r#k#ident);
    two_tokens!(r#k ident);
    three_tokens!(r#k#r);
    three_tokens!(r#k#r#ident);

    // Inserting a whitespace gets you the previous lexing behaviour
    three_tokens!(k# ident);
    three_tokens!(k #ident);
    three_tokens!(k # ident);

    // These should work anyway
    two_tokens!(k#);
    two_tokens!(k #);
    three_tokens!(k#123);
    three_tokens!(k#"normal string");
    three_tokens!(k#'c');
    three_tokens!(k#'lifetime);
    three_tokens!(k#[doc("doc attribute")]);

    #[cfg(False)]
    mod inside_cfg_false {
        const k#SOMETHING: i32 = 42;
        //~^ ERROR raw keyword syntax is reserved for future use
    }

    fn returns_k() -> i32 {
        let k = 42;
        k#k
        //~^ ERROR raw keyword syntax is reserved for future use
    }


    // Test that the token counting macros actually count the right amount of tokens.
    two_tokens!(k#knampf);
    //~^ ERROR unexpected end of macro invocation
    //~^^ ERROR raw keyword syntax is reserved for future use
    two_tokens!(k # knampf);
    //~^ ERROR no rules expected the token `knampf`
}
