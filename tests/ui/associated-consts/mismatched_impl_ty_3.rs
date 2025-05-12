//@ run-pass
trait Trait {
    const ASSOC: for<'a, 'b> fn(&'a u32, &'b u32);
}
impl Trait for () {
    const ASSOC: for<'a> fn(&'a u32, &'a u32) = |_, _| ();
}

fn main() {
    let _ = <() as Trait>::ASSOC;
}
