#![warn(clippy::single_char_lifetime_names)]

// Lifetimes should only be linted when they're introduced
struct DiagnosticCtx<'a, 'b>
where
    'a: 'b,
{
    _source: &'a str,
    _unit: &'b (),
}

// Only the lifetimes on the `impl`'s generics should be linted
impl<'a, 'b> DiagnosticCtx<'a, 'b> {
    fn new(source: &'a str, unit: &'b ()) -> DiagnosticCtx<'a, 'b> {
        Self {
            _source: source,
            _unit: unit,
        }
    }
}

// No lifetimes should be linted here
impl<'src, 'unit> DiagnosticCtx<'src, 'unit> {
    fn new_pass(source: &'src str, unit: &'unit ()) -> DiagnosticCtx<'src, 'unit> {
        Self {
            _source: source,
            _unit: unit,
        }
    }
}

// Only 'a should be linted here
fn split_once<'a>(base: &'a str, other: &'_ str) -> (&'a str, Option<&'a str>) {
    base.split_once(other)
        .map(|(left, right)| (left, Some(right)))
        .unwrap_or((base, None))
}

fn main() {
    let src = "loop {}";
    let unit = ();
    DiagnosticCtx::new(src, &unit);
}
