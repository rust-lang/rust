#![feature(fn_traits)]
#![feature(adt_const_params)]
//~^ WARNING the feature `adt_const_params` is incomplete

#[derive(PartialEq, Eq)]
struct CompileTimeSettings{
    hooks: &'static[fn()],
}

struct Foo<const T: CompileTimeSettings>;
//~^ ERROR using function pointers as const generic parameters is forbidden

impl<const T: CompileTimeSettings> Foo<T> {
    //~^ ERROR using function pointers as const generic parameters is forbidden
    fn call_hooks(){
    }
}

fn main(){
    const SETTINGS: CompileTimeSettings = CompileTimeSettings{
        hooks: &[],
    };

    Foo::<SETTINGS>::call_hooks();
}
