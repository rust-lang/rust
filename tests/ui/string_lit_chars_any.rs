//@aux-build:proc_macros.rs
#![warn(clippy::string_lit_chars_any)]
#![expect(clippy::eq_op, clippy::no_effect)]

#[macro_use]
extern crate proc_macros;

struct NotStringLit;

impl NotStringLit {
    fn chars(&self) -> impl Iterator<Item = char> {
        "c".chars()
    }
}

fn main() {
    let c = 'c';
    "\\.+*?()|[]{}^$#&-~".chars().any(|x| x == c);
    //~^ string_lit_chars_any
    r#"\.+*?()|[]{}^$#&-~"#.chars().any(|x| x == c);
    //~^ string_lit_chars_any
    "\\.+*?()|[]{}^$#&-~".chars().any(|x| c == x);
    //~^ string_lit_chars_any
    r#"\.+*?()|[]{}^$#&-~"#.chars().any(|x| c == x);
    //~^ string_lit_chars_any
    #[rustfmt::skip]
    "\\.+*?()|[]{}^$#&-~".chars().any(|x| { x == c });
    //~^ string_lit_chars_any
    // Do not lint
    NotStringLit.chars().any(|x| x == c);
    "\\.+*?()|[]{}^$#&-~".chars().any(|x| {
        let c = 'c';
        x == c
    });
    "\\.+*?()|[]{}^$#&-~".chars().any(|x| {
        1;
        x == c
    });
    "\\.+*?()|[]{}^$#&-~".chars().any(|x| x == x);
    "\\.+*?()|[]{}^$#&-~".chars().any(|_x| c == c);
    matches!(
        c,
        '\\' | '.' | '+' | '*' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$' | '#' | '&' | '-' | '~'
    );
    external! {
        let c = 'c';
        "\\.+*?()|[]{}^$#&-~".chars().any(|x| x == c);
    }
    with_span! {
        span
        let c = 'c';
        "\\.+*?()|[]{}^$#&-~".chars().any(|x| x == c);
    }
}
