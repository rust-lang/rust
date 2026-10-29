// Issue: 103366 , Suggest fix for misplaced generic params
// The generics fail to parse here, so don't make any suggestions/help

#[allow(unused)]
fn<~>()> id(x: T) -> T { x }
//~^ ERROR expected identifier, found `<`

fn main() {}
