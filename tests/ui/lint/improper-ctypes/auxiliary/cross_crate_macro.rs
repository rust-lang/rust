#[macro_export]
macro_rules! make_extern_fn {
    () => {
        extern "C" fn bad(p: ::std::string::String) {}
    };
}
