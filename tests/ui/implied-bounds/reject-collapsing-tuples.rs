// Rejection: collapsing nested reference structure inside a tuple
// The source has (&'a &'b (),) implying 'b: 'a; the target loses the nested ref.

fn foo<'a, 'b>(_: (&'a &'b (),), v: &'b u32) -> &'a u32 { v }

fn main() {
    let _f: for<'x> fn((&'x (),), &'x u32) -> &'x u32 = foo;
    //~^ ERROR mismatched types
}
