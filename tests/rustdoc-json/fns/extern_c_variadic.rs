#![feature(no_core)]
#![no_core]

extern "C" {
    // @is "$.index[*][?(@.name == 'not_variadic')].inner.decl.c_variadic" false
    pub fn not_variadic(_: i32);
    // @is "$.index[*][?(@.name == 'variadic')].inner.decl.c_variadic" true
    pub fn variadic(_: i32, ...);
}
