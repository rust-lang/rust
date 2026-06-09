use super::BackendTypes;

pub trait AbiBuilderMethods: BackendTypes {
    fn get_param(&mut self, index: usize) -> Self::Value;
}
