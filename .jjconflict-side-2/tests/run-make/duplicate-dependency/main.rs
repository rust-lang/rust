struct Bar;

impl From<Bar> for foo::Foo {
    fn from(_: Bar) -> Self {
        foo::Foo
    }
}

fn main() {
    // The user might wrongly expect this to work since From<Bar> for Foo
    // implies Into<Foo> for Bar. What the user missed is that different
    // versions of Foo exist in the dependency graph, and the impl is for the
    // wrong version.
    re_export_foo::into_foo(Bar);
}
