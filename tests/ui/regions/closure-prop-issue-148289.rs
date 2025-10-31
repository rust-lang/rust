//@ check-pass

#[derive(Clone, Copy)]
struct Inv<'a>(*mut &'a ());
impl<'a> Inv<'a> {
    fn outlived_by<'b: 'a>(self, _: Inv<'b>) {}
}
struct OutlivedBy<'a, 'b: 'a>(Inv<'a>, Inv<'b>);

fn closure_arg<'b, 'c, 'd>(
    _: impl for<'a> FnOnce(Inv<'a>, OutlivedBy<'a, 'b>, OutlivedBy<'a, 'c>, Inv<'d>),
) {
}
fn foo<'b, 'c, 'd: 'b>() {
    closure_arg::<'b, 'c, 'd>(|a, b, c, d| {
        a.outlived_by(b.1);
        a.outlived_by(c.1);
        b.1.outlived_by(d);
    });
}

fn main() {}
