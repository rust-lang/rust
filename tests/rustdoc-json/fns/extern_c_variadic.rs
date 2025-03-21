extern "C" {
    //@ is "$.index[?(@.name == 'not_variadic')].inner.function.sig.is_c_variadic" false
    pub fn not_variadic(_: i32);
    //@ is "$.index[?(@.name == 'variadic')].inner.function.sig.is_c_variadic" true
    pub fn variadic(_: i32, ...);
}
