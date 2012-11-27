// A curious inner-module that's not exported that contains the binding
// 'std' so that macro-expanded references to std::serialization and such
// can be resolved within libcore.
#[doc(hidden)] // FIXME #3538
mod std {
    pub use serialization;
}
