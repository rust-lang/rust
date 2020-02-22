// run-rustfix
trait TraitB {
    type Item;
}

trait TraitA<A> {
    type Type;
    fn bar<T>(_: T) -> Self;
    fn baz<T>(_: T) -> Self where T: TraitB, <T as TraitB>::Item: Copy;
}

struct S;
struct Type;

impl TraitA<()> for S { //~ ERROR not all trait items implemented
}

fn main() {}
