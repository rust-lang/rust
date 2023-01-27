trait ValidTrait {}

/// This has docs
pub fn f() -> impl ValidTrait {
    Vec::<DoesNotExist>::new()
    //~^ ERROR: cannot find type `DoesNotExist` in this scope
}
