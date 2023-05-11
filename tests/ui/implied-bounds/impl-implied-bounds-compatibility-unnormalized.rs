#![deny(implied_bounds_entailment)]

trait Project {
    type Ty;
}
impl Project for &'_ &'_ () {
    type Ty = ();
}
trait Trait {
    fn get<'s>(s: &'s str, _: ()) -> &'static str;
}
impl Trait for () {
    fn get<'s>(s: &'s str, _: <&'static &'s () as Project>::Ty) -> &'static str {
        //~^ ERROR impl method assumes more implied bounds than the corresponding trait method
        //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        s
    }
}
fn main() {
    let val = <() as Trait>::get(&String::from("blah blah blah"), ());
    println!("{}", val);
}
