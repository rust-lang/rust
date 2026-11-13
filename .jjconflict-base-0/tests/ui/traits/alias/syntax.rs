//@ check-pass

#![feature(trait_alias)]

trait SimpleAlias = Default;
trait GenericAlias<T> = Iterator<Item = T>;
trait Partial<T> = IntoIterator<Item = T>;
trait SpecificAlias = GenericAlias<i32>;
trait PartialEqRef<'a, T: 'a> = PartialEq<&'a T>;
trait StaticAlias = 'static;

trait Things<T> {}
trait Romeo {}
#[allow(dead_code)]
struct The<T>(T);
#[allow(dead_code)]
struct Fore<T>(T);
impl<T, U> Things<T> for The<U> {}
impl<T> Romeo for Fore<T> {}

trait WithWhere<Art, Thou> = Romeo + Romeo where Fore<(Art, Thou)>: Romeo;
trait BareWhere<Wild, Are> = where The<Wild>: Things<Are>;

fn main() {}
