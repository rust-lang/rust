use super::BackendTypes;

// FIXME(eddyb) move this into either `BuilderMethods` or some other abstraction
// meant to handle whole-function aspects like params and stack slots (`alloca`s).
pub trait AbiBuilderMethods: BackendTypes {
    fn get_param(&self, index: usize) -> Self::Value;
}
