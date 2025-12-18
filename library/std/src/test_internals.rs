/// A collection of common re-exports to be used by the test version of this crate.
pub use crate::thread::current::CURRENT as CURRENT_THREAD;

cfg_select! {
    target_thread_local => {
        pub use crate::thread::current::id::ID as CURRENT_THREAD_ID;
    }
    target_pointer_width = "16" => {
        pub use crate::thread::current::id::ID0 as CURRENT_THREAD_ID0;
        pub use crate::thread::current::id::ID16 as CURRENT_THREAD_ID16;
        pub use crate::thread::current::id::ID32 as CURRENT_THREAD_ID32;
        pub use crate::thread::current::id::ID48 as CURRENT_THREAD_ID48;
    }
    target_pointer_width = "32" => {
        pub use crate::thread::current::id::ID0 as CURRENT_THREAD_ID0;
        pub use crate::thread::current::id::ID32 as CURRENT_THREAD_ID32;
    }
    _ => {
        pub use crate::thread::current::id::ID as CURRENT_THREAD_ID;
    }
}
