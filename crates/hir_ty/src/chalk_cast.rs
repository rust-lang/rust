//! Implementations of the Chalk `Cast` trait for our types.

use chalk_ir::{
    cast::{Cast, CastTo},
    interner::HasInterner,
};

use crate::{AliasEq, DomainGoal, GenericArg, GenericArgData, Interner, TraitRef, Ty, WhereClause};

macro_rules! has_interner {
    ($t:ty) => {
        impl HasInterner for $t {
            type Interner = crate::Interner;
        }
    };
}

has_interner!(WhereClause);
has_interner!(DomainGoal);
has_interner!(GenericArg);
has_interner!(Ty);

impl CastTo<WhereClause> for TraitRef {
    fn cast_to(self, _interner: &Interner) -> WhereClause {
        WhereClause::Implemented(self)
    }
}

impl CastTo<WhereClause> for AliasEq {
    fn cast_to(self, _interner: &Interner) -> WhereClause {
        WhereClause::AliasEq(self)
    }
}

impl CastTo<DomainGoal> for WhereClause {
    fn cast_to(self, _interner: &Interner) -> DomainGoal {
        DomainGoal::Holds(self)
    }
}

impl CastTo<GenericArg> for Ty {
    fn cast_to(self, interner: &Interner) -> GenericArg {
        GenericArg::new(interner, GenericArgData::Ty(self))
    }
}

macro_rules! transitive_impl {
    ($a:ty, $b:ty, $c:ty) => {
        impl CastTo<$c> for $a {
            fn cast_to(self, interner: &Interner) -> $c {
                self.cast::<$b>(interner).cast(interner)
            }
        }
    };
}

// In Chalk, these can be done as blanket impls, but that doesn't work here
// because of coherence

transitive_impl!(TraitRef, WhereClause, DomainGoal);
transitive_impl!(AliasEq, WhereClause, DomainGoal);

macro_rules! reflexive_impl {
    ($a:ty) => {
        impl CastTo<$a> for $a {
            fn cast_to(self, _interner: &Interner) -> $a {
                self
            }
        }
    };
}

reflexive_impl!(GenericArg);
