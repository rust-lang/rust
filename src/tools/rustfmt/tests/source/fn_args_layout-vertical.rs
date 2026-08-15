// rustfmt-fn_params_layout: Vertical

// Empty list should stay on one line.
fn do_bar(

) -> u8 {
    bar()
}

// A single argument should stay on the same line.
fn do_bar(
        a: u8) -> u8 {
    bar()
}

// Multiple arguments should each get their own line.
fn do_bar(a: u8, mut b: u8, c: &u8, d: &mut u8, closure: &Fn(i32) -> i32) -> i32 {
    // This feature should not affect closures.
    let bar = |x: i32, y: i32| -> i32 { x + y };
    bar(a, b)
}

// If the first argument doesn't fit on the same line with the function name,
// the whole list should probably be pushed to the next line with hanging
// indent. That's not what happens though, so check current behaviour instead.
// In any case, it should maintain single argument per line.
fn do_this_that_and_the_other_thing(
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: u8,
        b: u8, c: u8, d: u8) {
    this();
    that();
    the_other_thing();
}
