#![warn(clippy::manual_ok_err)]

fn funcall() -> Result<u32, &'static str> {
    todo!()
}

fn main() {
    let _ = match funcall() {
        //~^ manual_ok_err
        Ok(v) => Some(v),
        Err(_) => None,
    };

    let _ = match funcall() {
        //~^ manual_ok_err
        Ok(v) => Some(v),
        _v => None,
    };

    let _ = match funcall() {
        //~^ manual_ok_err
        Err(v) => Some(v),
        Ok(_) => None,
    };

    let _ = match funcall() {
        //~^ manual_ok_err
        Err(v) => Some(v),
        _v => None,
    };

    let _ = if let Ok(v) = funcall() {
        //~^ manual_ok_err
        Some(v)
    } else {
        None
    };

    let _ = if let Err(v) = funcall() {
        //~^ manual_ok_err
        Some(v)
    } else {
        None
    };

    #[allow(clippy::redundant_pattern)]
    let _ = match funcall() {
        //~^ manual_ok_err
        Ok(v) => Some(v),
        _v @ _ => None,
    };

    struct S;

    impl std::ops::Neg for S {
        type Output = Result<u32, &'static str>;

        fn neg(self) -> Self::Output {
            funcall()
        }
    }

    // Suggestion should be properly parenthesized
    let _ = match -S {
        //~^ manual_ok_err
        Ok(v) => Some(v),
        _ => None,
    };

    no_lint();
}

fn no_lint() {
    let _ = match funcall() {
        Ok(v) if v > 3 => Some(v),
        _ => None,
    };

    let _ = match funcall() {
        Err(_) => None,
        Ok(3) => None,
        Ok(v) => Some(v),
    };

    let _ = match funcall() {
        _ => None,
        Ok(v) => Some(v),
    };

    let _ = match funcall() {
        Err(_) | Ok(3) => None,
        Ok(v) => Some(v),
    };

    #[expect(clippy::redundant_pattern)]
    let _ = match funcall() {
        _v @ _ => None,
        Ok(v) => Some(v),
    };

    // Content of `Option` and matching content of `Result` do
    // not have the same type.
    let _: Option<&dyn std::any::Any> = match Ok::<_, ()>(&1) {
        Ok(v) => Some(v),
        _ => None,
    };

    let _ = match Ok::<_, ()>(&1) {
        _x => None,
        Ok(v) => Some(v),
    };

    let _ = match Ok::<_, std::convert::Infallible>(1) {
        Ok(3) => None,
        Ok(v) => Some(v),
    };
}

const fn cf(x: Result<u32, &'static str>) -> Option<u32> {
    // Do not lint in const code
    match x {
        Ok(v) => Some(v),
        Err(_) => None,
    }
}
