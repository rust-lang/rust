fn main() {
    (0..)
        .map(
            #[target_feature(enable = "")]
            //~^ ERROR: the feature named `` is not valid for this target
            //~| ERROR: `#[target_feature(..)]` can only be applied to `unsafe` functions
            #[track_caller]
            //~^ ERROR: `#[track_caller]` requires Rust ABI
            |_| (),
        )
        .next();
}
