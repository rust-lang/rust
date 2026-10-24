trait Trait {
    type P<T: Copy, U: Copy>;
}
impl Trait for () {
    type P<T: Copy, U: Copy> = ();
}
fn main() {
    let _: <() as Trait>::P<String, String>;
    //~^ ERROR the trait bound `String: Copy` is not satisfied
}
