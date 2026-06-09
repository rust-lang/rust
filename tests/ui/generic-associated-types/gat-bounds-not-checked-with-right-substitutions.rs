// This test checks that we correctly reject the following unsound code.

trait Lengthen<T> {
    fn lengthen(self) -> T;
}

impl<'a> Lengthen<&'a str> for &'a str {
    fn lengthen(self) -> &'a str { self }
}

trait Gat {
    type Gat<'a>: for<'b> Lengthen<Self::Gat<'b>>;

    fn lengthen(s: Self::Gat<'_>) -> Self::Gat<'static> {
        s.lengthen()
    }
}

impl Gat for () {
    type Gat<'a> = &'a str; //~ ERROR: implementation of `Lengthen` is not general enough
}

fn main() {
    let s = "hello, garbage".to_string();
    let borrow: &'static str = <() as Gat>::lengthen(&s);
    drop(s);

    println!("{borrow}");
}
