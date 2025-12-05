//! Check that if the parser suggests converting `///` to a regular comment
//! when it appears after a missing comma in an list (e.g. `enum` variants).
//!
//! Related issue
//! - https://github.com/rust-lang/rust/issues/142311

enum Foo {
    /// Like the noise a sheep makes
    Bar
    /// Like where people drink
    //~^ ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found doc comment `/// Like where people drink`
    Baa///xxxxxx
    //~^ ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found doc comment `///xxxxxx`
    Baz///xxxxxx
    //~^ ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found doc comment `///xxxxxx`
}

fn foo() {
    let a = [
        1///xxxxxx
        //~^ ERROR expected one of `,`, `.`, `;`, `?`, `]`, or an operator, found doc comment `///xxxxxx`
        2
    ];
}

fn bar() {
    let a = [
        1,
        2///xxxxxx
        //~^ ERROR expected one of `,`, `.`, `?`, `]`, or an operator, found doc comment `///xxxxxx`
    ];
}

fn main() {}
