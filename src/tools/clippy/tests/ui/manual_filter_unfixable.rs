//@no-rustfix: the suggestion drops the closure's comment, so it is not machine-applicable

#![warn(clippy::manual_filter)]

fn issue17376(opt: Option<u32>) {
    // Rewriting to `filter` would drop the comment below, so the suggestion must not be
    // machine-applicable here. Otherwise `clippy --fix` would silently eat the comment (#17376).
    opt.and_then(|x| {
        //~^ manual_filter
        // keep this explanation
        if x < 10 { Some(x) } else { None }
    });
}

fn main() {}
