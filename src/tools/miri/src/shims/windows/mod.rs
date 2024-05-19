pub mod foreign_items;

mod env;
mod handle;
mod sync;
mod thread;

pub use env::WindowsEnvVars;
// All the Windows-specific extension traits
pub use env::EvalContextExt as _;
pub use handle::EvalContextExt as _;
pub use sync::EvalContextExt as _;
pub use thread::EvalContextExt as _;
