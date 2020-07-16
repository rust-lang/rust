trait ValidTrait {}

/// This has docs
pub fn f() -> impl ValidTrait {
    Vec::<DoesNotExist>::new()
    //~^ ERROR failed to resolve
}
