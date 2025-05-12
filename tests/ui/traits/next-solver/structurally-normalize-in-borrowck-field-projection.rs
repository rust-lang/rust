//@ check-pass
//@ compile-flags: -Znext-solver

trait Interner: Sized {
    type Value;
}

enum Kind<I: Interner> {
    Value(I::Value),
}

struct Intern;

impl Interner for Intern {
    type Value = Wrap<u32>;
}

struct Wrap<T>(T);

type KindAlias = Kind<Intern>;

trait PrettyPrinter: Sized {
    fn hello(c: KindAlias) {
        match c {
            KindAlias::Value(Wrap(v)) => {
                println!("{v:?}");
            }
        }
    }
}

fn main() {}
