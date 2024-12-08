//@ run-pass

trait ConstDefault {
    const DEFAULT: Self;
}

#[derive(PartialEq)]
enum E {
    V(i32),
    W(usize),
}

impl ConstDefault for E {
    const DEFAULT: Self = Self::V(23);
}

impl ConstDefault for Option<i32> {
    const DEFAULT: Self = Self::Some(23);
}

impl E {
    const NON_DEFAULT: Self = Self::W(12);
    const fn local_fn() -> Self {
        Self::V(23)
    }
}

const fn explicit_qpath() -> E {
    let _x = <Option<usize>>::Some(23);
    <E>::W(12)
}

fn main() {
    assert!(E::DEFAULT == E::local_fn());
    assert!(Option::DEFAULT == Some(23));
    assert!(E::NON_DEFAULT == explicit_qpath());
}
