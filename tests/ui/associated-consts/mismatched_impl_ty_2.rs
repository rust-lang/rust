//@ run-pass
trait Trait {
    const ASSOC: fn(&'static u32);
}
impl Trait for () {
    const ASSOC: for<'a> fn(&'a u32) = |_| ();
}

fn main() {
    let _ = <() as Trait>::ASSOC;
}
