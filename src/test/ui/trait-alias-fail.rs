// gate-test-trait_alias

trait Alias1<T> = Default where T: Clone; // ok
    //~^ERROR trait aliases are not yet fully implemented
trait Alias2<T: Clone = ()> = Default;
    //~^ERROR type parameters on the left side of a trait alias cannot be bounded
    //~^^ERROR type parameters on the left side of a trait alias cannot have defaults
    //~^^^ERROR trait aliases are not yet fully implemented

impl Alias1 { //~ERROR expected type, found trait alias
}

impl Alias1 for () { //~ERROR expected trait, found trait alias
}

fn main() {}

