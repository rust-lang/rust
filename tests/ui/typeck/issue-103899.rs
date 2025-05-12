//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)

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
    //~^ ERROR the trait bound `(): BaseWithAssoc` is not satisfied [E0277]
    loop {}
}

fn main() {}
