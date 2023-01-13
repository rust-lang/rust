trait TraitA<T> {
    fn func();
}

struct StructA {}

impl TraitA<i32> for StructA {
    fn func() {}
}

fn main() {
    TraitA::<i32>::func();
    //~^ ERROR: cannot call associated function on trait without specifying the corresponding `impl` type [E0790]
    //~| help: use the fully-qualified path to the only available implementation
}
