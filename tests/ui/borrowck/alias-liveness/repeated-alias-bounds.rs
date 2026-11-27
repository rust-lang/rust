//@ check-pass

// When visiting aliases for liveness, we should not do repeated work.

trait Left<T> {}
trait Right<T> {}

trait Def<'a> {
    type A0;
    type A1: Left<Self::A0> + Right<Self::A0>;
    type A2: Left<Self::A1> + Right<Self::A1>;
    type A3: Left<Self::A2> + Right<Self::A2>;
    type A4: Left<Self::A3> + Right<Self::A3>;
    type A5: Left<Self::A4> + Right<Self::A4>;
    type A6: Left<Self::A5> + Right<Self::A5>;
    type A7: Left<Self::A6> + Right<Self::A6>;
    type A8: Left<Self::A7> + Right<Self::A7>;
    type A9: Left<Self::A8> + Right<Self::A8>;
    type A10: Left<Self::A9> + Right<Self::A9>;
    type A11: Left<Self::A10> + Right<Self::A10>;
    type A12: Left<Self::A11> + Right<Self::A11>;
    type A13: Left<Self::A12> + Right<Self::A12>;
    type A14: Left<Self::A13> + Right<Self::A13>;
    type A15: Left<Self::A14> + Right<Self::A14>;
    type A16: Left<Self::A15> + Right<Self::A15>;
    type A17: Left<Self::A16> + Right<Self::A16>;
    type A18: Left<Self::A17> + Right<Self::A17>;
    type A19: Left<Self::A18> + Right<Self::A18>;
    type A20: Left<Self::A19> + Right<Self::A19>;
    type A21: Left<Self::A20> + Right<Self::A20>;
    type A22: Left<Self::A21> + Right<Self::A21>;
}

fn make<'a, T: Def<'a>>(_: &'a ()) -> T::A22 {
    loop {}
}

fn check<T: for<'a> Def<'a>>() {
    let x = ();
    let val = make::<T>(&x);
    drop(val);
}

fn main() {}
