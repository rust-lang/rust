#![feature(trait_alias)]

// Ensure that relaxed bounds are not permitted in the `Self` bounds of trait aliases because trait
// aliases (like traits) aren't implicitly bounded by `Sized` so there's nothing to relax.

trait Alias0 = ?Sized; //~ ERROR relaxed bounds are not permitted in trait alias bounds
trait Alias1 = where Self: ?Sized; //~ ERROR this relaxed bound is not permitted here

trait Alias2<T: ?Sized> =; // OK
trait Alias3<T> = where T: ?Sized; // OK

// Make sure that we don't permit "relaxing" trait aliases since we don't want to expand trait
// aliases during sized elaboration for simplicity as we'd need to handle relaxing arbitrary bounds
// (e.g., ones with modifiers, outlives-bounds, …) and where-clauses.

trait SizedAlias = Sized;
fn take<T: ?SizedAlias>() {} //~ ERROR bound modifier `?` can only be applied to `Sized`

fn main() {}
