//@check-pass
#![warn(clippy::match_like_matches_macro)]
#![feature(if_let_guard)]

#[expect(clippy::option_option)]
fn issue15841(opt: Option<Option<Option<i32>>>, value: i32) {
    let _ = match opt {
        Some(first)
            if let Some(second) = first
                && let Some(third) = second
                && third == value =>
        {
            true
        },
        _ => false,
    };

    // if-let is the second if
    let _ = match opt {
        Some(first)
            if first.is_some()
                && let Some(second) = first =>
        {
            true
        },
        _ => false,
    };

    // if-let is the third if
    let _ = match opt {
        Some(first)
            if first.is_some()
                && first.is_none()
                && let Some(second) = first =>
        {
            true
        },
        _ => false,
    };

    // don't get confused by `or`s
    let _ = match opt {
        Some(first)
            if (first.is_some() || first.is_none())
                && let Some(second) = first =>
        {
            true
        },
        _ => false,
    };
}
