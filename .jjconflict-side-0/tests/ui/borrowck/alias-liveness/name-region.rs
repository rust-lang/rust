// Make sure we don't ICE when trying to name the regions that appear in the alias
// of the type test error.

trait AnotherTrait {
    type Ty2<'a>;
}

fn test_alias<T: AnotherTrait>(_: &'static T::Ty2<'_>) {
    let _: &'static T::Ty2<'_>;
    //~^ ERROR the associated type `<T as AnotherTrait>::Ty2<'_>` may not live long enough
}

fn main() {}
