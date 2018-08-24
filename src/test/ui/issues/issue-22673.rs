trait Expr : PartialEq<Self::Item> {
    //~^ ERROR: cycle detected
    type Item;
}

fn main() {}
