#![feature(trait_alias)]

trait SimpleAlias = Default; //~ERROR E0645
trait GenericAlias<T> = Iterator<Item=T>; //~ERROR E0645
trait Partial<T> = IntoIterator<Item=T>; //~ERROR E0645

trait Things<T> {}
trait Romeo {}
struct The<T>(T);
struct Fore<T>(T);
impl<T, U> Things<T> for The<U> {}
impl<T> Romeo for Fore<T> {}

trait WithWhere<Art, Thou> = Romeo + Romeo where Fore<(Art, Thou)>: Romeo; //~ERROR E0645
trait BareWhere<Wild, Are> = where The<Wild>: Things<Are>; //~ERROR E0645

trait CD = Clone + Default; //~ERROR E0645

fn foo<T: CD>() -> (T, T) {
    let one = T::default();
    let two = one.clone();
    (one, two)
}

fn main() {
    let both = foo();
    assert_eq!(both.0, 0);
    assert_eq!(both.1, 0);
    let both: (i32, i32) = foo();
    assert_eq!(both.0, 0);
    assert_eq!(both.1, 0);
}

