// https://github.com/rust-lang/rust/issues/65462
//@ build-pass

enum Empty {}
enum Enum {
    Empty( Empty )
}

fn foobar() -> Option< Enum > {
    let value: Option< Empty > = None;
    Some( Enum::Empty( value? ) )
}

fn main() {
    foobar();
}
