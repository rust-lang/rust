//@ check-pass

trait SomeTrait {}
trait OtherTrait {
    type Item;
}

trait ErrorSimpleExample {
    type AssociatedType: SomeTrait;
    type GatBounded<T: SomeTrait>;
    type ErrorMinimal: OtherTrait<Item = Self::GatBounded<Self::AssociatedType>>;
}

fn main() {}
