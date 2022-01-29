// check-pass

trait Expr: PartialEq<Self::Item> {
    type Item;
}

fn main() {}
