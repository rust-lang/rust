// Issue: 103366 , Suggest fix for misplaced generic params
// run-rustfix

#[allow(unused)]
enum<T> Foo { Variant(T) }
//~^ ERROR expected identifier, found `<`
//~| HELP place the generic parameter name after the enum name

fn main() {}
