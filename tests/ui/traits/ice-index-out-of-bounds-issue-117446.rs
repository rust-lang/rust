//@ check-fail
//
// Regression for https://github.com/rust-lang/rust/issues/117446

pub struct Repeated<T>(Vec<T>);

trait Foo<'a> {
    fn outer<D>() -> Option<()>;
}

impl<'a, T> Foo<'a> for Repeated<T> {
    fn outer() -> Option<()> {
        //~^ ERROR  associated function `outer` has 0 type parameters but its trait declaration has 1 type parameter [E0049]
        //~^^ ERROR mismatched types [E0308]
        fn inner<Q>(value: Option<()>) -> Repeated<Q> {
            match value {
                _ => Self(unimplemented!()),
                //~^ ERROR can't reference `Self` constructor from outer item [E0401]
            }
        }
    }
}

fn main() {}
