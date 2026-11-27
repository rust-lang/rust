#![feature(macro_attr)]

macro_rules! local_attr {
    attr() { $($body:tt)* } => {
        compile_error!(concat!("local_attr: ", stringify!($($body)*)));
    };
    //~^^ ERROR: local_attr
}

//~v NOTE: `fn_only` exists, but has no `attr` rules
macro_rules! fn_only {
    {} => {}
}

//~v NOTE: `attr_only` exists, but has no rules for function-like invocation
macro_rules! attr_only {
    attr() {} => {}
}

fn main() {
    //~v NOTE: in this expansion of #[local_attr]
    #[local_attr]
    struct S;

    //~vv ERROR: cannot find macro `local_attr` in this scope
    //~| NOTE: `local_attr` is in scope, but it is an attribute
    local_attr!(arg);

    //~v ERROR: cannot find attribute `fn_only` in this scope
    #[fn_only]
    struct S;

    attr_only!(); //~ ERROR: cannot find macro `attr_only` in this scope
}

//~vv ERROR: cannot find attribute `forward_referenced_attr` in this scope
//~| NOTE: consider moving the definition of `forward_referenced_attr` before this call
#[forward_referenced_attr]
struct S;

//~v NOTE: a macro with the same name exists, but it appears later
macro_rules! forward_referenced_attr {
    attr() {} => {}
}

//~vv ERROR: cannot find attribute `cyclic_attr` in this scope
//~| NOTE: consider moving the definition of `cyclic_attr` before this call
#[cyclic_attr]
//~v NOTE: a macro with the same name exists, but it appears later
macro_rules! cyclic_attr {
    attr() {} => {}
}

macro_rules! attr_with_safety {
    unsafe attr() { struct RequiresUnsafe; } => {};
    attr() { struct SafeInvocation; } => {};
}

#[attr_with_safety]
struct SafeInvocation;

//~v ERROR: unnecessary `unsafe` on safe attribute invocation
#[unsafe(attr_with_safety)]
struct SafeInvocation;

//~v ERROR: unsafe attribute invocation requires `unsafe`
#[attr_with_safety]
struct RequiresUnsafe;

#[unsafe(attr_with_safety)]
struct RequiresUnsafe;
