// https://github.com/rust-lang/rust/issues/56229
//@ check-pass

trait Mirror {
    type Other;
}

#[derive(Debug)]
struct Even(usize);
struct Odd;

impl Mirror for Even {
    type Other = Odd;
}

impl Mirror for Odd {
    type Other = Even;
}

trait Dyn<T: Mirror>: AsRef<<T as Mirror>::Other> {}

impl Dyn<Odd> for Even {}

impl AsRef<Even> for Even {
    fn as_ref(&self) -> &Even {
        self
    }
}

fn code<T: Mirror>(d: &dyn Dyn<T>) -> &T::Other {
    d.as_ref()
}

fn main() {
    println!("{:?}", code(&Even(22)));
}
