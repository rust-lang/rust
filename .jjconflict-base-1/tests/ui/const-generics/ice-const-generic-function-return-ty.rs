// #95163
fn return_ty() -> impl Into<<() as Reexported;
//~^ ERROR expected one of `(`, `::`, `<`, or `>`, found `;`

fn main() {}
