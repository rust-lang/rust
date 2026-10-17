//@ run-rustfix

struct GenericAssocMethod<T>(T);

impl<T> GenericAssocMethod<T> {
    fn default_hello() {}
    fn self_ty_hello(_: Self) {}
    fn self_ty_ref_hello(_: &Self) {}
}

fn main() {
    // Test for inferred types
    let x = GenericAssocMethod(33);
    x.self_ty_ref_hello();
    //~^ ERROR no method named `self_ty_ref_hello` found
    x.self_ty_hello();
    //~^ ERROR no method named `self_ty_hello` found
    // Test for known types
    let y = GenericAssocMethod(33i32);
    y.default_hello();
    //~^ ERROR no method named `default_hello` found
    y.self_ty_ref_hello();
    //~^ ERROR no method named `self_ty_ref_hello` found
    y.self_ty_hello();
    //~^ ERROR no method named `self_ty_hello` found
}
