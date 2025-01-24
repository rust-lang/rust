#![allow(unused)]
#![deny(explicit_outlives_requirements)]


// These examples should live in edition-lint-infer-outlives.rs, but are split
// into this separate file because they can't be `rustfix`'d (and thus, can't
// be part of a `run-rustfix` test file) until rust-lang-nursery/rustfix#141
// is solved

mod structs {
    use std::fmt::Debug;

    struct TeeOutlivesAyIsDebugBee<'a, 'b, T: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T
    }

    struct TeeWhereOutlivesAyIsDebugBee<'a, 'b, T> where T: 'a + Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T
    }

    struct TeeYooOutlivesAyIsDebugBee<'a, 'b, T, U: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: T,
        yoo: &'a &'b U
    }

    struct TeeOutlivesAyYooBeeIsDebug<'a, 'b, T: 'a, U: 'b + Debug> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    struct TeeOutlivesAyYooIsDebugBee<'a, 'b, T: 'a, U: Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    struct TeeOutlivesAyYooWhereBee<'a, 'b, T: 'a, U> where U: 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    struct TeeYooWhereOutlivesAyIsDebugBee<'a, 'b, T, U> where U: 'a + Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: T,
        yoo: &'a &'b U
    }

    struct TeeOutlivesAyYooWhereBeeIsDebug<'a, 'b, T: 'a, U> where U: 'b + Debug {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    struct TeeOutlivesAyYooWhereIsDebugBee<'a, 'b, T: 'a, U> where U: Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    struct TeeWhereOutlivesAyYooWhereBeeIsDebug<'a, 'b, T, U> where T: 'a, U: 'b + Debug {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    struct TeeWhereOutlivesAyYooWhereIsDebugBee<'a, 'b, T, U> where T: 'a, U: Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    struct BeeOutlivesAyTeeBee<'a, 'b: 'a, T: 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T,
    }

    struct BeeOutlivesAyTeeAyBee<'a, 'b: 'a, T: 'a + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T,
    }

    struct BeeOutlivesAyTeeOutlivesAyIsDebugBee<'a, 'b: 'a, T: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T
    }

    struct BeeWhereAyTeeWhereOutlivesAyIsDebugBee<'a, 'b, T> where 'b: 'a, T: 'a + Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T
    }

    struct BeeOutlivesAyTeeYooOutlivesAyIsDebugBee<'a, 'b: 'a, T, U: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: T,
        yoo: &'a &'b U
    }

    struct BeeWhereAyTeeYooWhereOutlivesAyIsDebugBee<'a, 'b, T, U>
        where U: 'a + Debug + 'b, 'b: 'a
        //~^ ERROR outlives requirements can be inferred
    {
        tee: T,
        yoo: &'a &'b U
    }
}

mod tuple_structs {
    use std::fmt::Debug;

    struct TeeOutlivesAyIsDebugBee<'a, 'b, T: 'a + Debug + 'b>(&'a &'b T);
    //~^ ERROR outlives requirements can be inferred

    struct TeeWhereOutlivesAyIsDebugBee<'a, 'b, T>(&'a &'b T) where T: 'a + Debug + 'b;
    //~^ ERROR outlives requirements can be inferred

    struct TeeYooOutlivesAyIsDebugBee<'a, 'b, T, U: 'a + Debug + 'b>(T, &'a &'b U);
    //~^ ERROR outlives requirements can be inferred

    struct TeeOutlivesAyYooBeeIsDebug<'a, 'b, T: 'a, U: 'b + Debug>(&'a T, &'b U);
    //~^ ERROR outlives requirements can be inferred

    struct TeeOutlivesAyYooIsDebugBee<'a, 'b, T: 'a, U: Debug + 'b>(&'a T, &'b U);
    //~^ ERROR outlives requirements can be inferred

    struct TeeOutlivesAyYooWhereBee<'a, 'b, T: 'a, U>(&'a T, &'b U) where U: 'b;
    //~^ ERROR outlives requirements can be inferred

    struct TeeYooWhereOutlivesAyIsDebugBee<'a, 'b, T, U>(T, &'a &'b U) where U: 'a + Debug + 'b;
    //~^ ERROR outlives requirements can be inferred

    struct TeeOutlivesAyYooWhereBeeIsDebug<'a, 'b, T: 'a, U>(&'a T, &'b U) where U: 'b + Debug;
    //~^ ERROR outlives requirements can be inferred

    struct TeeOutlivesAyYooWhereIsDebugBee<'a, 'b, T: 'a, U>(&'a T, &'b U) where U: Debug + 'b;
    //~^ ERROR outlives requirements can be inferred

    struct TeeWhereAyYooWhereBeeIsDebug<'a, 'b, T, U>(&'a T, &'b U) where T: 'a, U: 'b + Debug;
    //~^ ERROR outlives requirements can be inferred

    struct TeeWhereAyYooWhereIsDebugBee<'a, 'b, T, U>(&'a T, &'b U) where T: 'a, U: Debug + 'b;
    //~^ ERROR outlives requirements can be inferred

    struct BeeOutlivesAyTeeBee<'a, 'b: 'a, T: 'b>(&'a &'b T);
    //~^ ERROR outlives requirements can be inferred

    struct BeeOutlivesAyTeeAyBee<'a, 'b: 'a, T: 'a + 'b>(&'a &'b T);
    //~^ ERROR outlives requirements can be inferred

    struct BeeOutlivesAyTeeOutlivesAyIsDebugBee<'a, 'b: 'a, T: 'a + Debug + 'b>(&'a &'b T);
    //~^ ERROR outlives requirements can be inferred

    struct BeeWhereAyTeeWhereAyIsDebugBee<'a, 'b, T>(&'a &'b T) where 'b: 'a, T: 'a + Debug + 'b;
    //~^ ERROR outlives requirements can be inferred

    struct BeeOutlivesAyTeeYooOutlivesAyIsDebugBee<'a, 'b: 'a, T, U: 'a + Debug + 'b>(T, &'a &'b U);
    //~^ ERROR outlives requirements can be inferred

    struct BeeWhereAyTeeYooWhereAyIsDebugBee<'a, 'b, T, U>(T, &'a &'b U)
        where U: 'a + Debug + 'b, 'b: 'a;
    //~^ ERROR outlives requirements can be inferred
}

mod enums {
    use std::fmt::Debug;

    enum TeeOutlivesAyIsDebugBee<'a, 'b, T: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        V { tee: &'a &'b T },
    }

    enum TeeWhereOutlivesAyIsDebugBee<'a, 'b, T> where T: 'a + Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        V(&'a &'b T),
    }

    enum TeeYooOutlivesAyIsDebugBee<'a, 'b, T, U: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        V { tee: T, },
        W(&'a &'b U),
    }

    enum TeeOutlivesAyYooBeeIsDebug<'a, 'b, T: 'a, U: 'b + Debug> {
        //~^ ERROR outlives requirements can be inferred
        V { tee: &'a T, yoo: &'b U },
        W,
    }

    enum TeeOutlivesAyYooIsDebugBee<'a, 'b, T: 'a, U: Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        V(&'a T, &'b U),
        W,
    }

    enum TeeOutlivesAyYooWhereBee<'a, 'b, T: 'a, U> where U: 'b {
        //~^ ERROR outlives requirements can be inferred
        V { tee: &'a T },
        W(&'b U),
    }

    enum TeeYooWhereOutlivesAyIsDebugBee<'a, 'b, T, U> where U: 'a + Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        V { tee: T, yoo: &'a &'b U },
        W,
    }

    enum TeeOutlivesAyYooWhereBeeIsDebug<'a, 'b, T: 'a, U> where U: 'b + Debug {
        //~^ ERROR outlives requirements can be inferred
        V(&'a T, &'b U),
        W,
    }

    enum TeeOutlivesAyYooWhereIsDebugBee<'a, 'b, T: 'a, U> where U: Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        V { tee: &'a T },
        W(&'b U)
    }

    enum TeeWhereOutlivesAyYooWhereBeeIsDebug<'a, 'b, T, U> where T: 'a, U: 'b + Debug {
        //~^ ERROR outlives requirements can be inferred
        V { tee: &'a T, yoo: &'b U },
        W,
    }

    enum TeeWhereOutlivesAyYooWhereIsDebugBee<'a, 'b, T, U> where T: 'a, U: Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        V(&'a T, &'b U),
        W,
    }

    enum BeeOutlivesAyTeeBee<'a, 'b: 'a, T: 'b> {
        //~^ ERROR outlives requirements can be inferred
        V { tee: &'a &'b T },
    }

    enum BeeOutlivesAyTeeAyBee<'a, 'b: 'a, T: 'a + 'b> {
        //~^ ERROR outlives requirements can be inferred
        V { tee: &'a &'b T },
        W,
    }

    enum BeeOutlivesAyTeeOutlivesAyIsDebugBee<'a, 'b: 'a, T: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        V { tee: &'a &'b T },
    }

    enum BeeWhereAyTeeWhereOutlivesAyIsDebugBee<'a, 'b, T> where 'b: 'a, T: 'a + Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        V(&'a &'b T),
    }

    enum BeeOutlivesAyTeeYooOutlivesAyIsDebugBee<'a, 'b: 'a, T, U: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        V { tee: T },
        W(&'a &'b U),
    }

    enum BeeWhereAyTeeYooWhereOutlivesAyIsDebugBee<'a, 'b, T, U> where U: 'a + Debug + 'b, 'b: 'a {
        //~^ ERROR outlives requirements can be inferred
        V { tee: T, yoo: &'a &'b U },
    }
}

mod unions {
    use std::fmt::Debug;

    union TeeOutlivesAyIsDebugBee<'a, 'b, T: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T
    }

    union TeeWhereOutlivesAyIsDebugBee<'a, 'b, T> where T: 'a + Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T
    }

    union TeeYooOutlivesAyIsDebugBee<'a, 'b, T, U: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: *const T,
        yoo: &'a &'b U
    }

    union TeeOutlivesAyYooBeeIsDebug<'a, 'b, T: 'a, U: 'b + Debug> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    union TeeOutlivesAyYooIsDebugBee<'a, 'b, T: 'a, U: Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    union TeeOutlivesAyYooWhereBee<'a, 'b, T: 'a, U> where U: 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    union TeeYooWhereOutlivesAyIsDebugBee<'a, 'b, T, U> where U: 'a + Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: *const T,
        yoo: &'a &'b U
    }

    union TeeOutlivesAyYooWhereBeeIsDebug<'a, 'b, T: 'a, U> where U: 'b + Debug {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    union TeeOutlivesAyYooWhereIsDebugBee<'a, 'b, T: 'a, U> where U: Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    union TeeWhereOutlivesAyYooWhereBeeIsDebug<'a, 'b, T, U> where T: 'a, U: 'b + Debug {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    union TeeWhereOutlivesAyYooWhereIsDebugBee<'a, 'b, T, U> where T: 'a, U: Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a T,
        yoo: &'b U
    }

    union BeeOutlivesAyTeeBee<'a, 'b: 'a, T: 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T,
    }

    union BeeOutlivesAyTeeAyBee<'a, 'b: 'a, T: 'a + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T,
    }

    union BeeOutlivesAyTeeOutlivesAyIsDebugBee<'a, 'b: 'a, T: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T
    }

    union BeeWhereAyTeeWhereOutlivesAyIsDebugBee<'a, 'b, T> where 'b: 'a, T: 'a + Debug + 'b {
        //~^ ERROR outlives requirements can be inferred
        tee: &'a &'b T
    }

    union BeeOutlivesAyTeeYooOutlivesAyIsDebugBee<'a, 'b: 'a, T, U: 'a + Debug + 'b> {
        //~^ ERROR outlives requirements can be inferred
        tee: *const T,
        yoo: &'a &'b U
    }

    union BeeWhereAyTeeYooWhereOutlivesAyIsDebugBee<'a, 'b, T, U> where U: 'a + Debug + 'b, 'b: 'a {
        //~^ ERROR outlives requirements can be inferred
        tee: *const T,
        yoo: &'a &'b U
    }
}

// https://github.com/rust-lang/rust/issues/106870
mod multiple_predicates_with_same_span {
    macro_rules! m {
        ($($name:ident)+) => {
            struct Inline<'a, $($name: 'a,)+>(&'a ($($name,)+));
            //~^ ERROR: outlives requirements can be inferred
            struct FullWhere<'a, $($name,)+>(&'a ($($name,)+)) where $($name: 'a,)+;
            //~^ ERROR: outlives requirements can be inferred
            struct PartialWhere<'a, $($name,)+>(&'a ($($name,)+)) where (): Sized, $($name: 'a,)+;
            //~^ ERROR: outlives requirements can be inferred
            struct Interleaved<'a, $($name,)+>(&'a ($($name,)+))
            where
                (): Sized,
                $($name: 'a, $name: 'a, )+ //~ ERROR: outlives requirements can be inferred
                $($name: 'a, $name: 'a, )+;
        }
    }
    m!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
}

fn main() {}
