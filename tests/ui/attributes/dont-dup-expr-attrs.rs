//@ check-pass
//
// During development of #124141 at one point expression on attributes were
// being duplicated and `m1` caused an exponential blowup that caused OOM.
// The number of recursive calls depends on the number of doc comments on the
// expr block. On each recursive call, the `#[allow(deprecated)]` attribute(s) on
// the `0` somehow get duplicated, resulting in 1, 2, 4, 8, ... identical
// attributes.
//
// After the fix, the code compiles quickly and normally.

macro_rules! m1 {
    ($(#[$meta:meta])* { $e:expr }) => {
        m1! { expr: { $e }, unprocessed: [$(#[$meta])*] }
    };

    (expr: { $e:expr }, unprocessed: [ #[$meta:meta] $($metas:tt)* ]) => {
        m1! { expr: { $e }, unprocessed: [ $($metas)* ] }
    };

    (expr: { $e:expr }, unprocessed: []) => {
        { $e }
    }
}

macro_rules! m2 {
    ($(#[$meta:meta])* { $e:stmt }) => {
        m2! { stmt: { $e }, unprocessed: [$(#[$meta])*] }
    };

    (stmt: { $e:stmt }, unprocessed: [ #[$meta:meta] $($metas:tt)* ]) => {
        m2! { stmt: { $e }, unprocessed: [ $($metas)* ] }
    };

    (stmt: { $e:stmt }, unprocessed: []) => {
        { $e }
    }
}

macro_rules! m3 {
    ($(#[$meta:meta])* { $e:item }) => {
        m3! { item: { $e }, unprocessed: [$(#[$meta])*] }
    };

    (item: { $e:item }, unprocessed: [ #[$meta:meta] $($metas:tt)* ]) => {
        m3! { item: { $e }, unprocessed: [ $($metas)* ] }
    };

    (item: { $e:item }, unprocessed: []) => {
        { $e }
    }
}

fn main() {
    // Each additional doc comment line doubles the compile time.
    m1!(
        /// a1
        /// a2
        /// a3
        /// a4
        /// a5
        /// a6
        /// a7
        /// a8
        /// a9
        /// a10
        /// a11
        /// a12
        /// a13
        /// a14
        /// a15
        /// a16
        /// a17
        /// a18
        /// a19
        /// a20
        {
            #[allow(deprecated)] 0
        }
    );

    m2!(
        /// a1
        /// a2
        /// a3
        /// a4
        /// a5
        /// a6
        /// a7
        /// a8
        /// a9
        /// a10
        /// a11
        /// a12
        /// a13
        /// a14
        /// a15
        /// a16
        /// a17
        /// a18
        /// a19
        /// a20
        {
            #[allow(deprecated)] let x = 5
        }
    );

    m3!(
        /// a1
        /// a2
        /// a3
        /// a4
        /// a5
        /// a6
        /// a7
        /// a8
        /// a9
        /// a10
        /// a11
        /// a12
        /// a13
        /// a14
        /// a15
        /// a16
        /// a17
        /// a18
        /// a19
        /// a20
        {
            #[allow(deprecated)] struct S;
        }
    );
}
