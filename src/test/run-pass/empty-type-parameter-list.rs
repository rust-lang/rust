// Test that empty type parameter list (<>) is synonymous with
// no type parameters at all

struct S<>;
trait T<> {}
enum E<> { V }
impl<> T<> for S<> {}
impl T for E {}
fn foo<>() {}
fn bar() {}

fn main() {
    let _ = S;
    let _ = S::<>;
    let _ = E::V;
    let _ = E::<>::V;
    foo();
    foo::<>();

    // Test that we can supply <> to non generic things
    bar::<>();
    let _: i32<>;
}
