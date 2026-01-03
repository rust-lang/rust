#![warn(clippy::equatable_if_let)]
#![allow(clippy::eq_op)]
#![feature(const_trait_impl, const_cmp)]

fn issue15376() {
    enum ConstEq {
        A,
        B,
    }
    impl const PartialEq for ConstEq {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }

    const C: ConstEq = ConstEq::A;

    // `impl PartialEq` is const... but we still suggest `matches!` for now
    // TODO: detect this and suggest `=`
    const _: u32 = if let ConstEq::A = C { 0 } else { 1 };
    //~^ ERROR: this pattern matching can be expressed using `matches!`
    const _: u32 = if let Some(ConstEq::A) = Some(C) { 0 } else { 1 };
    //~^ ERROR: this pattern matching can be expressed using `matches!`
}
