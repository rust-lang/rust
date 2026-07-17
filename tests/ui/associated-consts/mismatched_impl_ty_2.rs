trait Trait {
    const ASSOC: fn(&'static u32);
}
impl Trait for () {
    const ASSOC: for<'a> fn(&'a u32) = |_| (); //~ ERROR const not compatible with trait
}

fn main() {
    let _ = <() as Trait>::ASSOC;
}
