//! Target dependent parameters needed for layouts

use base_db::Crate;
use hir_def::layout::TargetDataLayout;
use rustc_abi::{AddressSpace, AlignFromBytesError, TargetDataLayoutErrors};
use triomphe::Arc;

use crate::db::HirDatabase;

pub fn target_data_layout_query(
    db: &dyn HirDatabase,
    krate: Crate,
) -> Result<Arc<TargetDataLayout>, Arc<str>> {
    match &krate.workspace_data(db).data_layout {
        Ok(it) => match TargetDataLayout::parse_from_llvm_datalayout_string(it, AddressSpace::ZERO) {
            Ok(it) => Ok(Arc::new(it)),
            Err(e) => {
                Err(match e {
                    TargetDataLayoutErrors::InvalidAddressSpace { addr_space, cause, err } => {
                        format!(
                            r#"invalid address space `{addr_space}` for `{cause}` in "data-layout": {err}"#
                        )
                    }
                    TargetDataLayoutErrors::InvalidBits { kind, bit, cause, err } => format!(r#"invalid {kind} `{bit}` for `{cause}` in "data-layout": {err}"#),
                    TargetDataLayoutErrors::MissingAlignment { cause } => format!(r#"missing alignment for `{cause}` in "data-layout""#),
                    TargetDataLayoutErrors::InvalidAlignment { cause, err } => format!(
                        r#"invalid alignment for `{cause}` in "data-layout": `{align}` is {err_kind}"#,
                        align = err.align(),
                        err_kind = match err {
                            AlignFromBytesError::NotPowerOfTwo(_) => "not a power of two",
                            AlignFromBytesError::TooLarge(_) => "too large",
                        }
                    ),
                    TargetDataLayoutErrors::InconsistentTargetArchitecture { dl, target } => {
                        format!(r#"inconsistent target specification: "data-layout" claims architecture is {dl}-endian, while "target-endian" is `{target}`"#)
                    }
                    TargetDataLayoutErrors::InconsistentTargetPointerWidth {
                        pointer_size,
                        target,
                    } => format!(r#"inconsistent target specification: "data-layout" claims pointers are {pointer_size}-bit, while "target-pointer-width" is `{target}`"#),
                    TargetDataLayoutErrors::InvalidBitsSize { err } => err,
                    TargetDataLayoutErrors::UnknownPointerSpecification { err } => format!(r#"use of unknown pointer specifer in "data-layout": {err}"#),
                }.into())
            }
        },
        Err(e) => Err(e.clone()),
    }
}
