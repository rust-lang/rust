// revisions: classic next
//[next] compile-flags: -Ztrait-solver=next

#![feature(non_lifetime_binders)]
//~^ WARNING the feature `non_lifetime_binders` is incomplete

fn take(id: impl for<T> Fn(T) -> T) {
    id(0);
    id("");
}

fn take2() -> impl for<T> Fn(T) -> T {
    //~^ ERROR expected a `Fn<(T,)>` closure, found
    //[classic]~| ERROR expected a `FnOnce<(T,)>` closure, found
    //[next]~| ERROR type mismatch resolving
    |x| x
}

fn main() {
    take(|x| x)
    //~^ ERROR expected a `Fn<(T,)>` closure, found
    //[classic]~| ERROR expected a `FnOnce<(T,)>` closure, found
    //[next]~| ERROR type mismatch resolving
}
