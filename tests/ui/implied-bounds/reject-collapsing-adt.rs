// Rejection: collapsing nested reference structure inside an ADT
// The source has Wrap<&'a &'b ()> implying 'b: 'a; the target loses the nested ref.

struct Wrap<T>(T);

fn foo<'a, 'b>(_: Wrap<&'a &'b ()>, v: &'b u32) -> &'a u32 { v }

fn main() {
    let _f: for<'x> fn(Wrap<&'x ()>, &'x u32) -> &'x u32 = foo;
    //~^ ERROR mismatched types
}
