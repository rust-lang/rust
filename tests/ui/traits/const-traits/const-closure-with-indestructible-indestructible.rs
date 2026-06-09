//@revisions: next old
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(const_trait_impl, const_closures, const_destruct)]
const fn i_need<F: [const] std::marker::Destruct>(x: F) {}

fn main() {
    const {
        let v = Vec::<u8>::new();
        i_need(const || {
            //~^ ERROR the trait bound
            let y = v;
        })
    };
}
