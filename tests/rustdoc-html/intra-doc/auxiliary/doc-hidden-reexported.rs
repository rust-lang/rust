#[doc(hidden)]
pub mod docs {
    /// Foo
    pub mod io_uring_provided_buffers {}
    /// Bar
    pub struct IoUringProvidedBuffers;
}

pub use docs::io_uring_provided_buffers;
pub use docs::IoUringProvidedBuffers;
