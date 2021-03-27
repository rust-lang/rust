macro builtin_functions($register:ident; $(fn $name:ident($($arg_name:ident: $arg_ty:ty),*) -> $ret_ty:ty;)*) {
    #[cfg(feature = "jit")]
    #[allow(improper_ctypes)]
    extern "C" {
        $(fn $name($($arg_name: $arg_ty),*) -> $ret_ty;)*
    }

    #[cfg(feature = "jit")]
    pub(crate) fn $register(builder: &mut cranelift_jit::JITBuilder) {
        for &(name, val) in &[$((stringify!($name), $name as *const u8)),*] {
            builder.symbol(name, val);
        }
    }
}

builtin_functions! {
    register_functions_for_jit;

    // integers
    fn __multi3(a: i128, b: i128) -> i128;
    fn __udivti3(n: u128, d: u128) -> u128;
    fn __divti3(n: i128, d: i128) -> i128;
    fn __umodti3(n: u128, d: u128) -> u128;
    fn __modti3(n: i128, d: i128) -> i128;
    fn __rust_u128_addo(a: u128, b: u128) -> (u128, bool);
    fn __rust_i128_addo(a: i128, b: i128) -> (i128, bool);
    fn __rust_u128_subo(a: u128, b: u128) -> (u128, bool);
    fn __rust_i128_subo(a: i128, b: i128) -> (i128, bool);
    fn __rust_u128_mulo(a: u128, b: u128) -> (u128, bool);
    fn __rust_i128_mulo(a: i128, b: i128) -> (i128, bool);

    // floats
    fn __floattisf(i: i128) -> f32;
    fn __floattidf(i: i128) -> f64;
    fn __floatuntisf(i: u128) -> f32;
    fn __floatuntidf(i: u128) -> f64;
    fn __fixsfti(f: f32) -> i128;
    fn __fixdfti(f: f64) -> i128;
    fn __fixunssfti(f: f32) -> u128;
    fn __fixunsdfti(f: f64) -> u128;
}
