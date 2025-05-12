// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Checks that code with exponential runtime does not have exponential behavior in inlining.

trait A {
    fn call();
}

trait B {
    fn call();
}
impl<T: A> B for T {
    #[inline]
    fn call() {
        <T as A>::call();
        <T as A>::call();
        <T as A>::call();
    }
}

trait C {
    fn call();
}
impl<T: B> C for T {
    #[inline]
    fn call() {
        <T as B>::call();
        <T as B>::call();
        <T as B>::call();
    }
}

trait D {
    fn call();
}
impl<T: C> D for T {
    #[inline]
    fn call() {
        <T as C>::call();
        <T as C>::call();
        <T as C>::call();
    }
}

trait E {
    fn call();
}
impl<T: D> E for T {
    #[inline]
    fn call() {
        <T as D>::call();
        <T as D>::call();
        <T as D>::call();
    }
}

trait F {
    fn call();
}
impl<T: E> F for T {
    #[inline]
    fn call() {
        <T as E>::call();
        <T as E>::call();
        <T as E>::call();
    }
}

trait G {
    fn call();
}
impl<T: F> G for T {
    #[inline]
    fn call() {
        <T as F>::call();
        <T as F>::call();
        <T as F>::call();
    }
}

impl A for () {
    #[inline(never)]
    fn call() {}
}

// EMIT_MIR exponential_runtime.main.Inline.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK-NOT: inlined
    // CHECK: (inlined <() as G>::call)
    // CHECK-NOT: inlined
    // CHECK: (inlined <() as F>::call)
    // CHECK-NOT: inlined
    // CHECK: (inlined <() as E>::call)
    // CHECK-NOT: inlined
    // CHECK: (inlined <() as D>::call)
    // CHECK-NOT: inlined
    // CHECK: (inlined <() as C>::call)
    // CHECK-NOT: inlined
    // CHECK: (inlined <() as B>::call)
    // CHECK-NOT: inlined
    <() as G>::call();
}
