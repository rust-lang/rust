// Issue: 103366 , Suggest fix for misplaced generic params
//@ run-rustfix

#[allow(unused)]
fn<'a, B: 'a + std::ops::Add<Output = u32>> f(_x: B) { }
//~^ ERROR expected identifier, found `<`
//~| HELP place the generic parameter name after the fn name

fn main() {}
