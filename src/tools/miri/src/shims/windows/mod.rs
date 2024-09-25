pub mod foreign_items;

mod env;
mod handle;
mod sync;
mod thread;

pub use self::env::WindowsEnvVars;
// All the Windows-specific extension traits
pub use self::env::EvalContextExt as _;
pub use self::handle::EvalContextExt as _;
pub use self::sync::EvalContextExt as _;
pub use self::thread::EvalContextExt as _;
