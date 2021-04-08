//! Implementations of the Chalk `Cast` trait for our types.

use chalk_ir::interner::HasInterner;

use crate::{CallableSig, ReturnTypeImplTraits};

macro_rules! has_interner {
    ($t:ty) => {
        impl HasInterner for $t {
            type Interner = crate::Interner;
        }
    };
}

has_interner!(CallableSig);
has_interner!(ReturnTypeImplTraits);
