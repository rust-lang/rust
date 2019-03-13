extern "C" {
    // @has variadic/fn.foo.html //pre 'pub unsafe extern "C" fn foo(x: i32, _: ...)'
    pub fn foo(x: i32, ...);
}
