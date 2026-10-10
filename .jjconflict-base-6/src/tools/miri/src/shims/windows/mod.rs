pub mod foreign_items;

mod env;
mod fs;
mod handle;
mod sync;
mod thread;

// All the Windows-specific extension traits
pub use self::env::{EvalContextExt as _, WindowsEnvVars};
pub use self::fs::EvalContextExt as _;
pub use self::handle::EvalContextExt as _;
pub use self::sync::EvalContextExt as _;
pub use self::thread::EvalContextExt as _;
