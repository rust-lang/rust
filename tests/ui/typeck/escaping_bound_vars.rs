// Test for issues/115517 which is fixed by pull/115486
// This should not ice
trait Test<const C: usize> {}

trait Elide<T> {
    fn call();
}

pub fn test()
where
    (): Test<{ 1 + (<() as Elide(&())>::call) }>,
    //~^ ERROR cannot capture late-bound lifetime in constant
    //~| ERROR associated type bindings are not allowed here
    //~| ERROR trait `Elide<(&(),)>` is not implemented for `()`
    //~| ERROR trait `Elide<(&(),)>` is not implemented for `()`
    //~| ERROR cannot add
{
}

fn main() {}
