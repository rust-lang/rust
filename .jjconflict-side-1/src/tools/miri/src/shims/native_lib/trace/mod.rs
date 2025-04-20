mod child;
pub mod messages;
mod parent;

pub use self::child::{Supervisor, init_sv, register_retcode_sv};

/// The size of the temporary stack we use for callbacks that the server executes in the client.
/// This should be big enough that `mempr_on` and `mempr_off` can safely be jumped into with the
/// stack pointer pointing to a "stack" of this size without overflowing it.
const CALLBACK_STACK_SIZE: usize = 1024;
