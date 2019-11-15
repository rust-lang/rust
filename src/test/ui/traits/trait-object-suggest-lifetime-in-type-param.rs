// compile-fail

trait DebugWith<Cx: ?Sized> {
    fn fmt_with(&self, cx: &Cx);
    fn debug_with(&self, cx: &Cx) {}
    fn debug_with_box(&self, cx: Box<&Cx>) {}
}


trait DebugContext {}

struct Foo {
    bar: Bar
}

impl DebugWith<dyn DebugContext> for Foo {
    fn fmt_with(&self, cx: &dyn DebugContext) {
        let Foo { bar } = self;
        bar.fmt_with(cx);
        //~^ ERROR cannot infer an appropriate lifetime
        bar.debug_with(cx);
        //~^ ERROR cannot infer an appropriate lifetime
    }

    fn debug_with_box(&self, cx: Box<&dyn DebugContext>) {
        let Foo { bar } = self;

        bar.debug_with_box(cx);
        //~^ ERROR mismatched types
    }
}

struct Bar {}

impl DebugWith<dyn DebugContext> for Bar {
    fn fmt_with(&self, cx: &dyn DebugContext) {}
}

fn main() {}
