// Regression test (rustc's own `rustc_attr_parsing`, `AcceptFn`/`AcceptContext`):
// reifying a fn item to a higher-ranked fn pointer whose argument type carries a
// *region-region* implied bound (`&'f mut Session<'sess>` implies `'sess: 'f`)
// must not fail with "higher-ranked subtype error". The bound is a genuine
// implied bound of the argument, so it is assumed during the reification's WF
// check.

//@ check-pass

struct Session<'sess>(&'sess ());

struct Context<'f, 'sess> {
    _sess: &'f mut Session<'sess>,
}

fn parse(_cx: &mut Context<'_, '_>) {}

type ParseFn = for<'sess> fn(&mut Context<'_, 'sess>);

fn reify() -> ParseFn {
    parse
}

fn main() {
    let _ = reify();
}
