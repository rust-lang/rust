trait Trait {
    type P<T: Copy, U: Copy>;
}
impl Trait for () {
    type P<T: Copy, U: Copy> = ();
}
fn main() {
    let _: <() as Trait>::P<String, String>;
    //~^ ERROR trait `Copy` is not implemented for `String`
}
