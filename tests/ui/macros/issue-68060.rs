fn main() {
    (0..)
        .map(
            #[target_feature(enable = "")]
            //~^ ERROR: attribute should be applied to a function
            //~| ERROR: feature named `` is not valid
            //~| NOTE: `` is not valid for this target
            #[track_caller]
            //~^ ERROR: `#[track_caller]` on closures is currently unstable
            //~| NOTE: see issue #87417
            |_| (),
            //~^ NOTE: not a function
        )
        .next();
}
