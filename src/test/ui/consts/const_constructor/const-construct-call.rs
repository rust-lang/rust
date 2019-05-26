// Test that constructors are considered to be const fns with the required feature.

// run-pass

// revisions: min_const_fn const_fn

#![cfg_attr(const_fn, feature(const_fn))]

#![feature(const_constructor)]

// Ctor(..) is transformed to Ctor { 0: ... } in HAIR lowering, so directly
// calling constructors doesn't require them to be const.

type ExternalType = std::panic::AssertUnwindSafe<(Option<i32>, Result<i32, bool>)>;

const fn call_external_constructors_in_local_vars() -> ExternalType {
    let f = Some;
    let g = Err;
    let h = std::panic::AssertUnwindSafe;
    let x = f(5);
    let y = g(false);
    let z = h((x, y));
    z
}

const CALL_EXTERNAL_CONSTRUCTORS_IN_LOCAL_VARS: ExternalType = {
    let f = Some;
    let g = Err;
    let h = std::panic::AssertUnwindSafe;
    let x = f(5);
    let y = g(false);
    let z = h((x, y));
    z
};

const fn call_external_constructors_in_temps() -> ExternalType {
    let x = { Some }(5);
    let y = (*&Err)(false);
    let z = [std::panic::AssertUnwindSafe][0]((x, y));
    z
}

const CALL_EXTERNAL_CONSTRUCTORS_IN_TEMPS: ExternalType = {
    let x = { Some }(5);
    let y = (*&Err)(false);
    let z = [std::panic::AssertUnwindSafe][0]((x, y));
    z
};

#[derive(Debug, PartialEq)]
enum LocalOption<T> {
    Some(T),
    _None,
}

#[derive(Debug, PartialEq)]
enum LocalResult<T, E> {
    _Ok(T),
    Err(E),
}

#[derive(Debug, PartialEq)]
struct LocalAssertUnwindSafe<T>(T);

type LocalType = LocalAssertUnwindSafe<(LocalOption<i32>, LocalResult<i32, bool>)>;

const fn call_local_constructors_in_local_vars() -> LocalType {
    let f = LocalOption::Some;
    let g = LocalResult::Err;
    let h = LocalAssertUnwindSafe;
    let x = f(5);
    let y = g(false);
    let z = h((x, y));
    z
}

const CALL_LOCAL_CONSTRUCTORS_IN_LOCAL_VARS: LocalType = {
    let f = LocalOption::Some;
    let g = LocalResult::Err;
    let h = LocalAssertUnwindSafe;
    let x = f(5);
    let y = g(false);
    let z = h((x, y));
    z
};

const fn call_local_constructors_in_temps() -> LocalType {
    let x = { LocalOption::Some }(5);
    let y = (*&LocalResult::Err)(false);
    let z = [LocalAssertUnwindSafe][0]((x, y));
    z
}

const CALL_LOCAL_CONSTRUCTORS_IN_TEMPS: LocalType = {
    let x = { LocalOption::Some }(5);
    let y = (*&LocalResult::Err)(false);
    let z = [LocalAssertUnwindSafe][0]((x, y));
    z
};

fn main() {
    assert_eq!(
        (
            call_external_constructors_in_local_vars().0,
            call_external_constructors_in_temps().0,
            call_local_constructors_in_local_vars(),
            call_local_constructors_in_temps(),
        ),
        (
            CALL_EXTERNAL_CONSTRUCTORS_IN_LOCAL_VARS.0,
            CALL_EXTERNAL_CONSTRUCTORS_IN_TEMPS.0,
            CALL_LOCAL_CONSTRUCTORS_IN_LOCAL_VARS,
            CALL_LOCAL_CONSTRUCTORS_IN_TEMPS,
        )
    );
}
