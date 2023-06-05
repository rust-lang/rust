// incremental

struct Wrapper<T>(T);

struct Local<T, U>(T, U);

impl<T> Local { //~ ERROR missing generics for struct `Local`
    type AssocType3 = T; //~ ERROR inherent associated types are unstable

    const WRAPPED_ASSOC_3: Wrapper<Self::AssocType3> = Wrapper();
}
//~^ ERROR `main` function not found
