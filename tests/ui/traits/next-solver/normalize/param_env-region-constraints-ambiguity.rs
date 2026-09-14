//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#265. Different where-clauses
// normalizing to the same bound caused ambiguity errors if we lazily normalized
// where-clauses when using them to prove a goal.
//
// We avoid these errors by eagerly normalizing the `param_env`.

mod one {
    trait Trait {
        type Assoc<'a>
        where
            Self: 'a;
    }

    trait Bound<'a> {}
    fn impls_bound<'a, T: Bound<'a>>() {}
    fn foo<'a, T: 'a>()
    where
        T: Trait<Assoc<'a> = T> + Bound<'a>,
        T::Assoc<'a>: Bound<'a>,
    {
        impls_bound::<'_, T>();
    }
}

mod two {
    trait Trait {
        type Assoc<'a>
        where
            Self: 'a;
    }

    trait Bound {}
    fn impls_bound<T: Bound>() {}
    fn foo<'a, T: 'a>()
    where
        T: Trait<Assoc<'a> = T> + Bound,
        T::Assoc<'a>: Bound,
    {
        impls_bound::<T>();
    }
}

// Minimization of tokio-par-util.
mod three {
    trait Trait1 {
        type Assoc1;
    }

    trait Trait2 {
        type Assoc2;
    }

    struct Indir<T>(T);
    impl<T> Trait2 for Indir<T>
    where
        T: Trait1,
        T::Assoc1: Trait2 + 'static,
    {
        type Assoc2 = <T::Assoc1 as Trait2>::Assoc2;
    }

    struct WrapperTwo<T, U>(T, U);

    impl<T, U> Trait1 for WrapperTwo<T, U>
    where
        T: Trait1,
        T::Assoc1: Trait2 + 'static,
        // additional region constraint in this candidate so they
        // can't be merged.
        U: Trait1<Assoc1 = <Indir<T> as Trait2>::Assoc2>,
        U: Trait1<Assoc1 = <T::Assoc1 as Trait2>::Assoc2>,
    {
        type Assoc1 = i32;
    }
}

// Minimization of `qazer`
mod four {
    trait Value {
        type SelfType<'a>
        where
            Self: 'a;
    }

    trait Repository<T> {}
    struct RedbRepo<From, Into>(From, Into);

    impl<From, Into> Repository<Into> for RedbRepo<From, Into>
    where
        for<'a> From: Value<SelfType<'a> = From> + Clone + 'static,
        for<'a> <From as Value>::SelfType<'a>: Clone,
    {
    }
}

fn main() {}
