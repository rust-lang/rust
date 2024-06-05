// We used to fatal error without any useful diagnostic when we had an unexpected
// token due to a strange interaction between the sequence parsing code and the
// param/lifetime parsing code.

fn hello() -> impl use<'a {}> Sized {}
//~^ ERROR expected one of `,` or `>`, found `{`

fn main() {}
