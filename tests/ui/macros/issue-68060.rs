fn main() {
    (0..)
        .map(
            #[target_feature(enable = "")]
            //~^ ERROR: attribute cannot be used on
            #[track_caller]
            //~^ ERROR: `#[track_caller]` on closures is currently unstable
            //~| NOTE: see issue #87417
            //~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
            |_| (),
        )
        .next();
}
