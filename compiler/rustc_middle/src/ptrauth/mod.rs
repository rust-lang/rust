pub mod discriminator;
pub mod llvm_siphash;

pub use discriminator::{
    FnPtrDiscriminatorSource, FnPtrTypeDiscriminatorInput, clone_discriminated_ptrauth_schema_for,
    compute_fn_ptr_type_discriminator_for,
};
