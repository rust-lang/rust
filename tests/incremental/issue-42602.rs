// Regression test for #42602. It used to be that we had
// a dep-graph like
//
//     typeck_root(foo) -> FnOnce -> typeck_root(bar)
//
// This was fixed by improving the resolution of the `FnOnce` trait
// selection node.

//@ revisions: bfail1 bfail2 bfail3
//@ compile-flags:-Zquery-dep-graph
//@ build-pass (FIXME(62277): could be check-pass?)
//@ ignore-backends: gcc

#![feature(rustc_attrs)]

fn main() {
    a::foo();
    b::bar();
}

mod a {
    #[cfg(bfail1)]
    pub fn foo() {
        let x = vec![1, 2, 3];
        let v = || ::std::mem::drop(x);
        v();
    }

    #[cfg(not(bfail1))]
    pub fn foo() {
        let x = vec![1, 2, 3, 4];
        let v = || ::std::mem::drop(x);
        v();
    }
}

mod b {
    #[rustc_clean(cfg="bfail2")]
    #[rustc_clean(cfg="bfail3")]
    pub fn bar() {
        let x = vec![1, 2, 3];
        let v = || ::std::mem::drop(x);
        v();
    }
}
