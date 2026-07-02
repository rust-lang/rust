pub mod discriminator;
pub mod llvm_siphash;

pub use discriminator::{
    FnPtrTypeDiscriminatorInput, build_fn_ptr_type_discriminator_input,
    compute_fn_ptr_type_discriminator,
};
