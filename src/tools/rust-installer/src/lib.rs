#[macro_use]
mod util;

mod combiner;
mod compression;
mod generator;
mod scripter;
mod tarballer;

pub use crate::combiner::Combiner;
pub use crate::generator::Generator;
pub use crate::scripter::Scripter;
pub use crate::tarballer::Tarballer;

/// The installer version, output only to be used by combine-installers.sh.
/// (should match `SOURCE_DIRECTORY/rust_installer_version`)
pub const RUST_INSTALLER_VERSION: u32 = 3;
