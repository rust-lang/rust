//! Things related to IR printing in the next-trait-solver.

use std::any::type_name_of_val;

use rustc_type_ir::inherent::SliceLike;
use rustc_type_ir::{self as ty, ir_print::IrPrint};

use super::SolverDefId;
use super::interner::DbInterner;

impl<'db> IrPrint<ty::AliasTy<Self>> for DbInterner<'db> {
    fn print(t: &ty::AliasTy<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(t: &ty::AliasTy<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::with_attached_db(|db| match t.def_id {
            SolverDefId::TypeAliasId(id) => fmt.write_str(&format!(
                "AliasTy({:?}[{:?}])",
                db.type_alias_signature(id).name.as_str(),
                t.args
            )),
            SolverDefId::InternedOpaqueTyId(id) => {
                fmt.write_str(&format!("AliasTy({:?}[{:?}])", id, t.args))
            }
            _ => panic!("Expected TypeAlias or OpaqueTy."),
        })
    }
}

impl<'db> IrPrint<ty::AliasTerm<Self>> for DbInterner<'db> {
    fn print(t: &ty::AliasTerm<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(t: &ty::AliasTerm<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::with_attached_db(|db| match t.def_id {
            SolverDefId::TypeAliasId(id) => fmt.write_str(&format!(
                "AliasTerm({:?}[{:?}])",
                db.type_alias_signature(id).name.as_str(),
                t.args
            )),
            SolverDefId::InternedOpaqueTyId(id) => {
                fmt.write_str(&format!("AliasTerm({:?}[{:?}])", id, t.args))
            }
            _ => panic!("Expected TypeAlias or OpaqueTy."),
        })
    }
}
impl<'db> IrPrint<ty::TraitRef<Self>> for DbInterner<'db> {
    fn print(t: &ty::TraitRef<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(t: &ty::TraitRef<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::with_attached_db(|db| {
            let trait_ = t.def_id.0;
            let self_ty = &t.args.as_slice()[0];
            let trait_args = &t.args.as_slice()[1..];
            if trait_args.is_empty() {
                fmt.write_str(&format!(
                    "{:?}: {}",
                    self_ty,
                    db.trait_signature(trait_).name.as_str()
                ))
            } else {
                fmt.write_str(&format!(
                    "{:?}: {}<{:?}>",
                    self_ty,
                    db.trait_signature(trait_).name.as_str(),
                    trait_args
                ))
            }
        })
    }
}
impl<'db> IrPrint<ty::TraitPredicate<Self>> for DbInterner<'db> {
    fn print(t: &ty::TraitPredicate<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(
        t: &ty::TraitPredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl<'db> IrPrint<rustc_type_ir::HostEffectPredicate<Self>> for DbInterner<'db> {
    fn print(
        t: &rustc_type_ir::HostEffectPredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(
        t: &rustc_type_ir::HostEffectPredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl<'db> IrPrint<ty::ExistentialTraitRef<Self>> for DbInterner<'db> {
    fn print(
        t: &ty::ExistentialTraitRef<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(
        t: &ty::ExistentialTraitRef<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        crate::with_attached_db(|db| {
            let trait_ = t.def_id.0;
            fmt.write_str(&format!(
                "ExistentialTraitRef({:?}[{:?}])",
                db.trait_signature(trait_).name.as_str(),
                t.args
            ))
        })
    }
}
impl<'db> IrPrint<ty::ExistentialProjection<Self>> for DbInterner<'db> {
    fn print(
        t: &ty::ExistentialProjection<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(
        t: &ty::ExistentialProjection<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        crate::with_attached_db(|db| {
            let id = match t.def_id {
                SolverDefId::TypeAliasId(id) => id,
                _ => panic!("Expected trait."),
            };
            fmt.write_str(&format!(
                "ExistentialProjection(({:?}[{:?}]) -> {:?})",
                db.type_alias_signature(id).name.as_str(),
                t.args,
                t.term
            ))
        })
    }
}
impl<'db> IrPrint<ty::ProjectionPredicate<Self>> for DbInterner<'db> {
    fn print(
        t: &ty::ProjectionPredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(
        t: &ty::ProjectionPredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        crate::with_attached_db(|db| {
            let id = match t.projection_term.def_id {
                SolverDefId::TypeAliasId(id) => id,
                _ => panic!("Expected trait."),
            };
            fmt.write_str(&format!(
                "ProjectionPredicate(({:?}[{:?}]) -> {:?})",
                db.type_alias_signature(id).name.as_str(),
                t.projection_term.args,
                t.term
            ))
        })
    }
}
impl<'db> IrPrint<ty::NormalizesTo<Self>> for DbInterner<'db> {
    fn print(t: &ty::NormalizesTo<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(
        t: &ty::NormalizesTo<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl<'db> IrPrint<ty::SubtypePredicate<Self>> for DbInterner<'db> {
    fn print(
        t: &ty::SubtypePredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(
        t: &ty::SubtypePredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl<'db> IrPrint<ty::CoercePredicate<Self>> for DbInterner<'db> {
    fn print(t: &ty::CoercePredicate<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(
        t: &ty::CoercePredicate<Self>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
impl<'db> IrPrint<ty::FnSig<Self>> for DbInterner<'db> {
    fn print(t: &ty::FnSig<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(t: &ty::FnSig<Self>, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}

impl<'db> IrPrint<rustc_type_ir::PatternKind<DbInterner<'db>>> for DbInterner<'db> {
    fn print(
        t: &rustc_type_ir::PatternKind<DbInterner<'db>>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Self::print_debug(t, fmt)
    }

    fn print_debug(
        t: &rustc_type_ir::PatternKind<DbInterner<'db>>,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fmt.write_str(&format!("TODO: {:?}", type_name_of_val(t)))
    }
}
