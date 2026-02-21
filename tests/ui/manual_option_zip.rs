#![warn(clippy::manual_option_zip)]
#![allow(clippy::bind_instead_of_map)]

fn main() {}

fn should_lint() {
    // basic case
    let a: Option<i32> = Some(1);
    let b: Option<i32> = Some(2);
    let _ = a.and_then(|a| b.map(|b| (a, b)));
    //~^ manual_option_zip

    // different types
    let a: Option<String> = Some(String::new());
    let b: Option<i32> = Some(1);
    let _ = a.and_then(|a| b.map(|b| (a, b)));
    //~^ manual_option_zip

    // with None receiver
    let b: Option<i32> = Some(2);
    let _ = None::<i32>.and_then(|a| b.map(|b| (a, b)));
    //~^ manual_option_zip

    // with function call as map receiver
    let a: Option<i32> = Some(1);
    let _ = a.and_then(|a| get_option().map(|b| (a, b)));
    //~^ manual_option_zip
}

fn should_not_lint() {
    let a: Option<i32> = Some(1);
    let b: Option<i32> = Some(2);

    // tuple order reversed: (inner, outer) instead of (outer, inner)
    let _ = a.and_then(|a| b.map(|b| (b, a)));

    // tuple has more than 2 elements
    let _ = a.and_then(|a| b.map(|b| (a, b, 1)));

    // inner closure body is not a simple tuple of the params
    let _ = a.and_then(|a| b.map(|b| (a, b + 1)));

    // map receiver uses the outer closure parameter
    let _ = a.and_then(|a| a.checked_add(1).map(|b| (a, b)));

    // not Option type (Result)
    let a: Result<i32, &str> = Ok(1);
    let _ = a.and_then(|a| Ok((a, 1)));

    // closure body is not a map call
    let a: Option<i32> = Some(1);
    let _ = a.and_then(|a| Some((a, 1)));

    // three-element tuple
    let _ = a.and_then(|a| b.map(|b| (a, b, a)));

    // single-element tuple
    let _ = a.and_then(|a| b.map(|_b| (a,)));

    // the outer param used in the map receiver (cannot extract)
    let opts: Vec<Option<i32>> = vec![Some(1), Some(2)];
    let _ = a.and_then(|a| opts[a as usize].map(|b| (a, b)));

    // n-ary zip where n > 2, which is out of scope for this lint (for now)
    let c: Option<i32> = Some(3);
    let _ = a.and_then(|a| b.and_then(|b| c.map(|c| (a, b, c))));
}

fn get_option() -> Option<i32> {
    Some(123)
}
