// This test is for regression of issue #145611
// There should not be cycle error in effective_visibilities query.

trait LocalTrait {}
struct SomeType;
fn impls_trait<T: LocalTrait>() {}
fn foo() -> impl Sized {
    impls_trait::<SomeType>(); //~ ERROR the trait bound `SomeType: LocalTrait` is not satisfied [E0277]
}

fn main() {}
