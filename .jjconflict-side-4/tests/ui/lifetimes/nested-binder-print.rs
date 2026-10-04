struct TwoLt<'a, 'b>(&'a (), &'b ());
type Foo<'a> = fn(TwoLt<'_, 'a>);

fn foo() {
    let y: for<'a> fn(Foo<'a>);
    let x: u32 = y;
    //~^ ERROR mismatched types
}

fn main() {}
