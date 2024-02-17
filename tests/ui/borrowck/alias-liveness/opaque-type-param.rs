//@ known-bug: #42940

trait Trait {}
impl Trait for () {}

fn foo<'a>(s: &'a str) -> impl Trait + 'static {
    bar(s)
}

fn bar<P: AsRef<str>>(s: P) -> impl Trait + 'static {
    ()
}

fn main() {}
