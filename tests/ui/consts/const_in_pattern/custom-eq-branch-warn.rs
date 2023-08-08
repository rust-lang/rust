// check-pass

struct CustomEq;

impl Eq for CustomEq {}
impl PartialEq for CustomEq {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

#[derive(PartialEq, Eq)]
enum Foo {
    Bar,
    Baz,
    Qux(CustomEq),
}

// We know that `BAR_BAZ` will always be `Foo::Bar` and thus eligible for structural matching, but
// dataflow will be more conservative.
const BAR_BAZ: Foo = if 42 == 42 {
    Foo::Bar
} else {
    Foo::Qux(CustomEq)
};

fn main() {
    match Foo::Qux(CustomEq) {
        BAR_BAZ => panic!(),
        //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
        //~| NOTE the traits must be derived
        //~| NOTE StructuralEq.html for details
        //~| WARN this was previously accepted
        //~| NOTE see issue #73448
        //~| NOTE `#[warn(nontrivial_structural_match)]` on by default
        _ => {}
    }
}
