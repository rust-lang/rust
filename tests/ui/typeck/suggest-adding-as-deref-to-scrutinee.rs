use std::io::ErrorKind;

type E = Vec<ErrorKind>;

fn res(res: Result<Vec<i32>, E>) {
    match res {
        //~^ HELP: consider using `as_deref` here
        Ok([1]) => true,
        //~^ ERROR: expected an array or slice
        Err(_) => false,
    };

    match res {
        //~^ HELP: consider using `as_deref` here
        Ok([1]) => true,
        //~^ ERROR: expected an array or slice
        Err([ErrorKind::NotFound]) => false,
        //~^ ERROR: expected an array or slice
    };
}

fn opt(opt: Option<Vec<i32>>) {
    match opt {
        //~^ HELP: consider using `as_deref` here
        Some([1]) => true,
        //~^ ERROR: expected an array or slice
        None => false,
    };
}

fn err_arm_only(res: Result<Vec<i32>, E>) {
    // doesn't suggest `as_deref` since the arm is `Err`,
    // which `as_deref` doesn't make sense
    match res {
        Err([ErrorKind::NotFound]) => true,
        //~^ ERROR: expected an array or slice
        _ => false,
    };
}

fn res_cannot_be_array_or_slice(res: Result<i32, E>) {
    // doesn't suggest `as_deref` since `T` in `Ok(T)` can't be dereferenced
    // to an array or slice
    match res {
        Ok([1]) => true,
        //~^ ERROR: expected an array or slice
        _ => false,
    };
}

fn opt_cannot_be_array_or_slice(opt: Option<i32>) {
    // doesn't suggest `as_deref` since `T` in `Some(T)` can't be dereferenced
    // to an array or slice
    match opt {
        Some([1]) => true,
        //~^ ERROR: expected an array or slice
        None => false,
    };
}

fn res_has_ref(res: Result<&Vec<i32>, E>) {
    // `T` in `Ok(T)` can't be dereferenced to an array or slice
    // because `res.as_deref()` would return `Result<&Vec<i32>, &E>`
    match res {
        Ok([1]) => true,
        //~^ ERROR: expected an array or slice
        Err(_) => false,
    };
}

fn opt_has_ref(opt: Option<&Vec<i32>>) {
    // `T` in `Some(T)` can't be dereferenced to an array or slice
    // because `opt.as_deref()` would return `Option<&Vec<i32>>`
    match opt {
        Some([1]) => true,
        //~^ ERROR: expected an array or slice
        None => false,
    };
}

fn main() {}
