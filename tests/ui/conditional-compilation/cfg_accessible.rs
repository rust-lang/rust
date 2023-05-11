#![feature(cfg_accessible)]

mod m {
    pub struct ExistingPublic;
    struct ExistingPrivate;
}

trait Trait {
    type Assoc;
}

enum Enum {
    Existing,
}

#[cfg_accessible(Enum)]
struct ExistingResolved;

#[cfg_accessible(Enum::Existing)]
struct ExistingResolvedVariant;

#[cfg_accessible(m::ExistingPublic)]
struct ExistingPublic;

#[cfg_accessible(m::ExistingPrivate)]
struct ExistingPrivate;

#[cfg_accessible(m::NonExistent)]
struct NonExistingPrivate;

#[cfg_accessible(n::AccessibleExpanded)] // OK, `cfg_accessible` can wait and retry.
struct AccessibleExpanded;

#[cfg_accessible(Trait::Assoc)]
struct AccessibleTraitAssoc;

macro_rules! generate_accessible_expanded {
    () => {
        mod n {
            pub struct AccessibleExpanded;
        }
    };
}

generate_accessible_expanded!();

fn main() {
    ExistingPublic;
    AccessibleExpanded;
    AccessibleTraitAssoc;

    ExistingPrivate; //~ ERROR cannot find
    NonExistingPrivate; //~ ERROR cannot find
    NonExistingTraitAlias; //~ ERROR cannot find
}
