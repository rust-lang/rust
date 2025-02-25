//@ check-pass

#![warn(returning_pointers_to_local_variables)]

fn foo() -> *const u32 {
    let empty = 42u32;
    return &empty as *const _;
    //~^ WARN returning a pointer to stack memory associated with a local variable
}

fn bar() -> *const u32 {
    let empty = 42u32;
    &empty as *const _
    //~^ WARN returning a pointer to stack memory associated with a local variable
}

fn baz() -> *const u32 {
    let empty = 42u32;
    return &empty;
    //~^ WARN returning a pointer to stack memory associated with a local variable
}

fn faa() -> *const u32 {
    let empty = 42u32;
    &empty
    //~^ WARN returning a pointer to stack memory associated with a local variable
}

fn pointer_to_pointer() -> *const *mut u32 {
    let mut empty = 42u32;
    &(&mut empty as *mut u32) as *const _
    //~^ WARN returning a pointer to stack memory associated with a local variable
}

fn dont_lint_param(val: u32) -> *const u32 {
    &val
}

struct Foo {}

impl Foo {
    fn dont_lint_self_param(&self) -> *const Foo {
        self
    }
}

fn main() {}
