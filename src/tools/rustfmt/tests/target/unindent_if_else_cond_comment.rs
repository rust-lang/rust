// Comments on else block. See #1575.

fn example() {
    // `if` comment
    if x {
        foo();
    // `else if` comment
    } else if y {
        foo();
    // Comment on `else if`.
    // Comment on `else if`.
    } else if z {
        bar();
    /*
     *  Multi line comment on `else if`
     */
    } else if xx {
        bar();
    /* Single line comment on `else if` */
    } else if yy {
        foo();
    // `else` comment
    } else {
        foo();
        // Comment at the end of `else` block
    };
}
