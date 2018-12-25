// FIXME: Investigate why expansion info for a single expansion id is reset from
// `MacroBang(format_args)` to `MacroAttribute(derive(Clone))` (issue #52363).

fn main() {
    format_args!({ #[derive(Clone)] struct S; });
    //~^ ERROR format argument must be a string literal
}
