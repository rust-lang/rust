// Regression test for #54779, checks if the diagnostics are confusing.

trait DebugWith<Cx: ?Sized> {
    fn debug_with<'me>(&'me self, cx: &'me Cx) -> DebugCxPair<'me, Self, Cx> {
        DebugCxPair { value: self, cx }
    }

    fn fmt_with(&self, cx: &Cx, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

struct DebugCxPair<'me, Value: ?Sized, Cx: ?Sized>
where
    Value: DebugWith<Cx>,
{
    value: &'me Value,
    cx: &'me Cx,
}

trait DebugContext {}

struct Foo {
    bar: Bar,
}

impl DebugWith<dyn DebugContext> for Foo {
    fn fmt_with(
        &self,
        cx: &dyn DebugContext,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let Foo { bar } = self;
        bar.debug_with(cx); //~ ERROR borrowed data escapes outside of method
        Ok(())
    }
}

struct Bar {}

impl DebugWith<dyn DebugContext> for Bar {
    fn fmt_with(
        &self,
        cx: &dyn DebugContext,
        fmt: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Ok(())
    }
}

fn main() {}
