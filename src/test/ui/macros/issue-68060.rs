fn main() {
    (0..)
        .map(
            #[target_feature(enable = "")]
            //~^ ERROR: attribute should be applied to a function
            //~| ERROR: the feature named `` is not valid for this target
            //~| NOTE: `` is not valid for this target
            #[track_caller]
            //~^ ERROR: attribute should be applied to function [E0739]
            //~| ERROR: `#[track_caller]` requires Rust ABI [E0737]
            |_| (),
            //~^ NOTE: not a function
            //~| NOTE: not a function
        )
        .next();
}
