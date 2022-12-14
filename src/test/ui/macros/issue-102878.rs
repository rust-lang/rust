macro_rules!test{($l:expr,$_:r)=>({const:y y)}
//~^ ERROR mismatched closing delimiter: `)`
//~| ERROR invalid fragment specifier `r`
//~| ERROR expected identifier, found keyword `const`
//~| ERROR expected identifier, found keyword `const`
//~| ERROR expected identifier, found `:`

fn s(){test!(1,i)}

fn main() {}
