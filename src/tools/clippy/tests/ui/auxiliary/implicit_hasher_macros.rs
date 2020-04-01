#[macro_export]
macro_rules! implicit_hasher_fn {
    () => {
        pub fn f(input: &HashMap<u32, u32>) {}
    };
}
