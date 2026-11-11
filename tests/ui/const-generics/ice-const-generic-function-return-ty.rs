// #95163
fn return_ty() -> impl Into<<() as Reexported;
//~^ ERROR expected one of `(`, `::`, `<`, or `>`, found `;`
//~^^ ERROR: free function without a body

fn main() {}
