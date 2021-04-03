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

macro_rules! transitive_impl {
    ($a:ty, $b:ty, $c:ty) => {
        impl CastTo<$c> for $a {
            fn cast_to(self, interner: &Interner) -> $c {
                self.cast::<$b>(interner).cast(interner)
            }
        }
    };
}
