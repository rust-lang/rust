//@ no-rustfix
#![warn(clippy::collapsible_if)]

fn issue13365() {
    // in the following examples, we won't lint because of the comments,
    // so the the `expect` will be unfulfilled
    if true {
        // don't collapsible because of this comment
        #[expect(clippy::collapsible_if)]
        if true {}
    }
    //~^^^ ERROR: this lint expectation is unfulfilled

    if true {
        #[expect(clippy::collapsible_if)]
        // don't collapsible because of this comment
        if true {}
    }
    //~^^^^ ERROR: this lint expectation is unfulfilled
}

#[allow(unexpected_cfgs)]
fn issue16715(o: Option<i32>) {
    if let Some(x) = o {
        if x > 0 {
            println!("Positive: {}", x);
        }

        #[cfg(feature = "some_feature")]
        {
            if x % 2 == 0 {
                println!("Even: {}", x);
            }
        }
    }
}
