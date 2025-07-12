extern "C" {
    //@ jq .index[] | select(.name == "not_variadic").inner.function.sig?.is_c_variadic == false
    pub fn not_variadic(_: i32);
    //@ jq .index[] | select(.name == "variadic").inner.function.sig?.is_c_variadic == true
    pub fn variadic(_: i32, ...);
}
