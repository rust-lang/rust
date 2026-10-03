pub enum TestEnum<
    T: std::collections::HashMap<String, Vec<Box<dyn std::fmt::Debug + Send + Sync + 'static>>> + Clone + Default + PartialEq + Eq + std::fmt::Debug + serde::Serialize + serde::Deserialize<'static> + Send + Sync + 'static = std::collections::HashMap<String, Vec<Box<dyn std::fmt::Debug + Send + Sync + 'static>>>,
> {
    Variant1(T),
    Variant2 { field: T },
}

// More realistic example from real codebase
pub enum ElementInit<
    P: wrt_foundation::MemoryProvider + Clone + Default + PartialEq + Eq = wrt_foundation::NoStdProvider<1024>,
> {
    FuncIndices(crate::WasmVec<u32, P>),
    Expressions(crate::WasmVec<crate::WasmVec<u8, P>, P>),
}
