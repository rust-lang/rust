#![feature(macro_derive)]

macro_rules! MyDerive {
    derive() { $($body:tt)* } => {
        compile_error!(concat!("MyDerive: ", stringify!($($body)*)));
    };
    //~^^ ERROR: MyDerive
}

macro_rules! fn_only {
//~^ NOTE: `fn_only` exists, but has no `derive` rules
//~| NOTE: `fn_only` exists, but has no `derive` rules
    {} => {}
}

//~v NOTE: `DeriveOnly` exists, but has no rules for function-like invocation
macro_rules! DeriveOnly {
    derive() {} => {}
}

fn main() {
    //~v NOTE: in this expansion of #[derive(MyDerive)]
    #[derive(MyDerive)]
    struct S1;

    //~vv ERROR: cannot find macro `MyDerive` in this scope
    //~| NOTE: `MyDerive` is in scope, but it is a derive
    MyDerive!(arg);

    #[derive(fn_only)]
    struct S2;
    //~^^ ERROR: cannot find derive macro `fn_only` in this scope
    //~| ERROR: cannot find derive macro `fn_only` in this scope
    //~| NOTE: duplicate diagnostic emitted

    DeriveOnly!(); //~ ERROR: cannot find macro `DeriveOnly` in this scope
}

#[derive(ForwardReferencedDerive)]
struct S;
//~^^ ERROR: cannot find derive macro `ForwardReferencedDerive` in this scope
//~| NOTE: consider moving the definition of `ForwardReferencedDerive` before this call
//~| ERROR: cannot find derive macro `ForwardReferencedDerive` in this scope
//~| NOTE: consider moving the definition of `ForwardReferencedDerive` before this call
//~| NOTE: duplicate diagnostic emitted

macro_rules! ForwardReferencedDerive {
//~^ NOTE: a macro with the same name exists, but it appears later
//~| NOTE: a macro with the same name exists, but it appears later
    derive() {} => {}
}
