use std::borrow::Cow;

use rustc_session::Session;

pub mod apple;
pub mod archive;
pub(crate) mod command;
pub mod link;
pub(crate) mod linker;
pub mod lto;
pub mod metadata;
pub(crate) mod rpath;
pub mod symbol_export;
pub mod write;

/// The target triple depends on the deployment target, and is required to
/// enable features such as cross-language LTO, and for picking the right
/// Mach-O commands.
///
/// Certain optimizations also depend on the deployment target.
pub fn versioned_llvm_target(sess: &Session) -> Cow<'_, str> {
    if sess.target.is_like_darwin {
        apple::add_version_to_llvm_target(&sess.target.llvm_target, sess.apple_deployment_target())
            .into()
    } else {
        // FIXME(madsmtm): Certain other targets also include a version,
        // we might want to move that here as well.
        Cow::Borrowed(&sess.target.llvm_target)
    }
}
