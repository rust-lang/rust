#![feature(rustc_attrs)]

pub struct NotAUse;

#[rustc_edition_redirect = "..=2024"]
//~^ ERROR the `rustc_edition_redirect` attribute cannot be used on structs
pub struct AlsoNotAUse;

mod source {
    pub struct Old;
    pub struct Current;
}

#[rustfmt::skip]
#[rustc_edition_redirect = "..=2024"]
//~^ ERROR `#[rustc_edition_redirect]` can only be applied to a single import
pub use source::{Current};
#[rustc_edition_redirect = "..=2024"]
//~^ ERROR `#[rustc_edition_redirect]` can only be applied to a single import
pub use source::*;

mod private {
    pub(crate) struct Old;
}

#[rustc_edition_redirect = "..=2024"]
//~^ ERROR edition redirect for `Public` must have the same visibility as its default item
pub use private::Old as Public;
//~^ ERROR `Old` is only public within the crate, and cannot be re-exported outside

pub struct Public;

#[rustc_edition_redirect = "..=2024"]
pub use source::Missing as Unresolved;
//~^ ERROR unresolved import `source::Missing`

pub struct Unresolved;

pub type OverlapTargetA = ();
pub type OverlapTargetB = ();

#[rustc_edition_redirect = "..=2021"]
pub use OverlapTargetA as Overlap;
#[rustc_edition_redirect = "2021..=2024"]
//~^ ERROR edition redirect range 2021..=2024 overlaps with another range for `Overlap`
pub use OverlapTargetB as Overlap;

pub type Overlap = ();

pub struct RestrictedTarget;

#[rustc_edition_redirect = "..=2024"]
//~^ ERROR edition redirect for `Restricted` must have the same visibility as its default item
pub(crate) use RestrictedTarget as Restricted;

pub struct Restricted;

pub struct MissingDefaultTarget;

#[rustc_edition_redirect = "..=2024"]
//~^ ERROR edition redirect for `MissingDefault` has no default item
pub use MissingDefaultTarget as MissingDefault;
#[rustc_edition_redirect = "not an edition"]
//~^ ERROR invalid edition range in edition redirect
pub use source::Old as InvalidEdition;
#[rustc_edition_redirect = "2024"]
//~^ ERROR invalid edition range in edition redirect
pub use source::Old as MissingRangeOperator;
#[rustc_edition_redirect = "2021.."]
//~^ ERROR invalid edition range in edition redirect
pub use source::Old as MissingInclusiveEnd;
#[rustc_edition_redirect = "..2021"]
//~^ ERROR invalid edition range in edition redirect
pub use source::Old as ExclusiveRange;
#[rustc_edition_redirect = "2024..=2021"]
//~^ ERROR invalid edition range in edition redirect
pub use source::Old as InvertedRange;

fn main() {}
