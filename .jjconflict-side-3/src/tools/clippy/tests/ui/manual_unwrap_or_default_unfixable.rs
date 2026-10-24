//@no-rustfix
fn issue_12670() {
    // no auto: type not found
    #[allow(clippy::match_result_ok)]
    let _ = if let Some(x) = "1".parse().ok() {
        //~^ manual_unwrap_or_default
        x
    } else {
        i32::default()
    };
    let _ = if let Some(x) = None { x } else { i32::default() };
    //~^ manual_unwrap_or_default
    // auto fix with unwrap_or_default
    let a: Option<i32> = None;
    let _ = if let Some(x) = a { x } else { i32::default() };
    //~^ manual_unwrap_or_default
    let _ = if let Some(x) = Some(99) { x } else { i32::default() };
    //~^ manual_unwrap_or_default
}
