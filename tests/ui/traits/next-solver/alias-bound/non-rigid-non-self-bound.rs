//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// A test showcasing that non-self item bounds are currently allowed to
// apply for the self-type only after normalization. We don't mark them
// as rigid.

mod via_item_bound {
    trait Bound {}

    trait Id {
        type This;
    }

    trait B {
        type Outer: Id<This = Self::Outer>;
    }

    trait A {
        type Inner: B<Outer: Id<This: Bound>>;
    }

    fn is_bound<T: Bound>() {}
    fn check<T: A>() {
        is_bound::<<T::Inner as B>::Outer>();
    }
}

mod via_env {
    trait Bound {}

    trait Id {
        type This;
    }
    impl<T> Id for T {
        type This = T;
    }

    trait B {
        type Outer;
    }

    trait A {
        type Inner: B<Outer: Id<This: Bound>>;
    }

    fn is_bound<T: Bound>() {}
    fn check<T: A>()
    where
        <T::Inner as B>::Outer: Id<This = <T::Inner as B>::Outer>,
    {
        is_bound::<<T::Inner as B>::Outer>();
    }
}

fn main() {}
