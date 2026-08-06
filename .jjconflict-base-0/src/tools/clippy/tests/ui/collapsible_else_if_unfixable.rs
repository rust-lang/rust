//@no-rustfix
#![warn(clippy::collapsible_else_if)]

fn issue_13365() {
    // in the following examples, we won't lint because of the comments,
    // so the the `expect` will be unfulfilled
    if true {
    } else {
        // some other text before
        #[expect(clippy::collapsible_else_if)]
        if false {}
    }
    //~^^^ ERROR: this lint expectation is unfulfilled

    if true {
    } else {
        #[expect(clippy::collapsible_else_if)]
        // some other text after
        if false {}
    }
    //~^^^^ ERROR: this lint expectation is unfulfilled
}
