// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

#![feature(rustc_attrs)]
#![feature(negative_impls)]

#[rustc_auto_trait]
unsafe trait Trait {
    type Output; //~ ERROR E0380
}

fn call_method<T: Trait>(x: T) {}

fn main() {
    // ICE
    call_method(());
}
