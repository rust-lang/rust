#![warn(clippy::elidable_lifetime_names)]

struct UnitVariantAccess<'a, 'b, 's>(&'a &'b &'s ());
trait Trait<'de> {}
impl<'de, 'a, 's> Trait<'de> for UnitVariantAccess<'a, 'de, 's> {}
//~^ elidable_lifetime_names
