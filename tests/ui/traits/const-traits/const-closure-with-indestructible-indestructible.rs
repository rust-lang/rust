//@revisions: next old
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(const_trait_impl, const_closures, const_destruct)]


struct NotConstDestruct;

impl Drop for NotConstDestruct {
    fn drop(&mut self) {}
}

const fn i_need<F: [const] core::marker::Destruct>(x: F) {}

fn main() {
    const {
        let n = NotConstDestruct;
        i_need(const || {
            //~^ ERROR the trait bound
            let y = n;
        })
    };
}
