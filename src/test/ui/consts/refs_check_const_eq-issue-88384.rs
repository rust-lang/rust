// check-pass

#![feature(fn_traits)]
#![feature(adt_const_params)]
//~^ WARNING the feature `adt_const_params` is incomplete

#[derive(PartialEq, Eq)]
struct CompileTimeSettings{
    hooks: &'static[fn()],
}

struct Foo<const T: CompileTimeSettings>;

impl<const T: CompileTimeSettings> Foo<T> {
    fn call_hooks(){
    }
}

fn main(){
    const SETTINGS: CompileTimeSettings = CompileTimeSettings{
        hooks: &[],
    };

    Foo::<SETTINGS>::call_hooks();
}
