use super::BackendTypes;

pub trait AbiBuilderMethods<'tcx>: BackendTypes {
    fn get_param(&mut self, index: usize) -> Self::Value;
}
