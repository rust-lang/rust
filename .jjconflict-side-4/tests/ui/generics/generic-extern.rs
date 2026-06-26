// Ensure that we reject generic parameters on foreign items.

extern "C" {
    fn foo<T>(); //~ ERROR foreign items may not have type parameters

    // Furthermore, check that type parameter defaults lead to a *hard* error,
    // not just a lint error, for maximum forward compatibility.
    #[allow(invalid_type_param_default)] // Should have no effect here.
    fn bar<T = ()>(); //~ ERROR foreign items may not have type parameters
    //~^ ERROR defaults for generic parameters are not allowed here
}

fn main() {
    unsafe { foo::<i32>() };
}
