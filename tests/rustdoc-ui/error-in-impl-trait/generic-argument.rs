//@ check-pass
trait ValidTrait {}

/// This has docs
pub fn f() -> impl ValidTrait {
    Vec::<DoesNotExist>::new()
}
