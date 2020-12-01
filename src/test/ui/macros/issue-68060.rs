fn main() {
    (0..)
        .map(
            #[target_feature(enable = "")]
            //~^ ERROR: attribute should be applied to a function
            #[track_caller]
            |_| (),
            //~^ NOTE: not a function
        )
        .next();
}
