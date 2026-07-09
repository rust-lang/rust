pub mod discriminator;
pub mod llvm_siphash;

pub use discriminator::{
    FnPtrTypeDiscriminatorInput, build_fn_ptr_type_discriminator_input_from_instance,
    build_fn_ptr_type_discriminator_input_from_sig, build_fn_ptr_type_discriminator_input_from_ty,
    compute_fn_ptr_type_discriminator,
};
