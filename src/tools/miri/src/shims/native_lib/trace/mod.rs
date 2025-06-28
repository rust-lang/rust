mod child;
pub mod messages;
mod parent;

pub use self::child::{Supervisor, init_sv, register_retcode_sv};

/// The size of the temporary stack we use for callbacks that the server executes in the client.
const CALLBACK_STACK_SIZE: usize = 1024;
