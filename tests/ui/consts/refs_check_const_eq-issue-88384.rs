#![feature(fn_traits)]
#![feature(adt_const_params, unsized_const_params)]
//~^ WARNING the feature `unsized_const_params` is incomplete

#[derive(PartialEq, Eq)]
struct CompileTimeSettings {
    hooks: &'static [fn()],
}

struct Foo<const T: CompileTimeSettings>;
//~^ ERROR `CompileTimeSettings` must implement `ConstParamTy` to be used as the type of a const generic parameter

impl<const T: CompileTimeSettings> Foo<T> {
    //~^ ERROR `CompileTimeSettings` must implement `ConstParamTy` to be used as the type of a const generic parameter
    fn call_hooks() {}
}

fn main() {
    const SETTINGS: CompileTimeSettings = CompileTimeSettings { hooks: &[] };

    Foo::<SETTINGS>::call_hooks();
}
