// Issue #97583.
trait Super<T> {
    type Assoc;
}
trait Trait: Super<u32> + Super<i32> {}

fn foo<T>(_: dyn Trait<Assoc = u32>, _: T) {}
//~^ ERROR: ambiguous associated type

fn main() {
    let _: dyn Trait<Assoc = u32>;
    //~^ ERROR: ambiguous associated type
}
