//! ABI-related things in the next-trait-solver.
use rustc_abi::ExternAbi;
use rustc_ast_ir::visit::VisitorResult;
use rustc_type_ir::{
    FallibleTypeFolder, TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor, error::TypeError,
    relate::Relate,
};

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

    fn unsafe_mode() -> Self {
        Safety::Unsafe
    }
}

impl<'db> TypeVisitable<DbInterner<'db>> for ExternAbi {
    fn visit_with<V: TypeVisitor<DbInterner<'db>>>(&self, _visitor: &mut V) -> V::Result {
        V::Result::output()
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for ExternAbi {
    fn try_fold_with<F: FallibleTypeFolder<DbInterner<'db>>>(
        self,
        _folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self)
    }

    fn fold_with<F: TypeFolder<DbInterner<'db>>>(self, _folder: &mut F) -> Self {
        self
    }
}

impl<'db> Relate<DbInterner<'db>> for ExternAbi {
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
