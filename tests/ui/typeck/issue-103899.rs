//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] check-fail
//@[current] failure-status: 101
//@[current] dont-check-compiler-stderr
//@[current] known-bug: #103899

trait BaseWithAssoc {
    type Assoc;
}

trait WrapperWithAssoc {
    type BaseAssoc: BaseWithAssoc;
}

struct Wrapper<B> {
    inner: B,
}

struct ProjectToBase<T: BaseWithAssoc> {
    data_type_h: T::Assoc,
}

struct DoubleProject<L: WrapperWithAssoc> {
    buffer: Wrapper<ProjectToBase<L::BaseAssoc>>,
}

fn trigger<L: WrapperWithAssoc<BaseAssoc = ()>>() -> DoubleProject<L> {
    loop {}
}

fn main() {}
