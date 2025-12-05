//! ABI-related things in the next-trait-solver.
use rustc_type_ir::{error::TypeError, relate::Relate};

use crate::FnAbi;

use super::interner::DbInterner;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Safety {
    Unsafe,
    Safe,
}

impl<'db> Relate<DbInterner<'db>> for Safety {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        _relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        if a != b {
            Err(TypeError::SafetyMismatch(rustc_type_ir::error::ExpectedFound::new(a, b)))
        } else {
            Ok(a)
        }
    }
}

impl<'db> rustc_type_ir::inherent::Safety<DbInterner<'db>> for Safety {
    fn safe() -> Self {
        Self::Safe
    }

    fn is_safe(self) -> bool {
        matches!(self, Safety::Safe)
    }

    fn prefix_str(self) -> &'static str {
        match self {
            Self::Unsafe => "unsafe ",
            Self::Safe => "",
        }
    }
}

impl<'db> Relate<DbInterner<'db>> for FnAbi {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        _relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        if a == b {
            Ok(a)
        } else {
            Err(TypeError::AbiMismatch(rustc_type_ir::error::ExpectedFound::new(a, b)))
        }
    }
}

impl<'db> rustc_type_ir::inherent::Abi<DbInterner<'db>> for FnAbi {
    fn rust() -> Self {
        FnAbi::Rust
    }

    fn is_rust(self) -> bool {
        // TODO: rustc does not consider `RustCall` to be true here, but Chalk does
        matches!(self, FnAbi::Rust | FnAbi::RustCall)
    }
}
