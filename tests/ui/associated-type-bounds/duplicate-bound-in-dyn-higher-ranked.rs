// Alpha-equivalent associated types are currently treated as different types
// for the purposes of prohibiting conflicting associated types in dyn.
// This is undesirable, but we don't have a good way of fixing this.
// See also https://github.com/rust-lang/rust/issues/146548

#![feature(trait_alias)]

trait Trait {
    type Assoc;
}
trait Alias = Trait<Assoc = for<'a> fn(&'a ())>;
fn via_trait_alias(_: &dyn Alias<Assoc = for<'b> fn(&'b ())>) {}
//~^ ERROR conflicting associated type bindings for `Assoc`

trait Super<T> {
    type Assoc;
}
trait Sub: Super<i32, Assoc = for<'a> fn(&'a ())> + Super<i64, Assoc = for<'b> fn(&'b ())> {}
fn via_supertrait(_: &dyn Sub) {}
//~^ ERROR conflicting associated type bindings for `Assoc`

struct Thing;
impl Trait for Thing {
    type Assoc = fn(&());
}
impl Super<i32> for Thing {
    type Assoc = fn(&());
}
impl Super<i64> for Thing {
    type Assoc = fn(&());
}
impl Sub for Thing {}

fn main() {
    via_trait_alias(&Thing);
    via_supertrait(&Thing);
}
