//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

// issue: rust-lang/rust#104779
// ICE region infer, IndexMap: key not found

struct Inv<'a>(&'a mut &'a ());
enum Foo<T> {
    Bar,
    Var(T),
}
type Subtype = Foo<for<'a, 'b> fn(Inv<'a>, Inv<'b>)>;
type Supertype = Foo<for<'a> fn(Inv<'a>, Inv<'a>)>;

fn foo() -> impl Sized {
//~^ WARN function cannot return without recursing
    loop {
        match foo() {
        //[current]~^ ERROR higher-ranked subtype error
        //[current]~^^ ERROR higher-ranked subtype error
            Subtype::Bar => (),
            //[current]~^ ERROR higher-ranked subtype error
            //[current]~^^ ERROR higher-ranked subtype error
            Supertype::Var(x) => {}
        }
    }
}

pub fn main() {}
