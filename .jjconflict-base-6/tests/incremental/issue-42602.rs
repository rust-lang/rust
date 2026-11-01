// Regression test for #42602. It used to be that we had
// a dep-graph like
//
//     typeck_root(foo) -> FnOnce -> typeck_root(bar)
//
// This was fixed by improving the resolution of the `FnOnce` trait
// selection node.

//@ revisions: bpass1 bpass2 bpass3
//@ compile-flags:-Zquery-dep-graph
//@ ignore-backends: gcc
// FIXME(#62277): could be check-pass?

#![feature(rustc_attrs)]

fn main() {
    a::foo();
    b::bar();
}

mod a {
    #[cfg(bpass1)]
    pub fn foo() {
        let x = vec![1, 2, 3];
        let v = || ::std::mem::drop(x);
        v();
    }

    #[cfg(not(bpass1))]
    pub fn foo() {
        let x = vec![1, 2, 3, 4];
        let v = || ::std::mem::drop(x);
        v();
    }
}

mod b {
    #[rustc_clean(cfg="bpass2")]
    #[rustc_clean(cfg="bpass3")]
    pub fn bar() {
        let x = vec![1, 2, 3];
        let v = || ::std::mem::drop(x);
        v();
    }
}
