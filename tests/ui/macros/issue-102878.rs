macro_rules!test{($l:expr,$_:r)=>({const:y y)}
//~^ ERROR mismatched closing delimiter: `)`

fn s(){test!(1,i)}

fn main() {}
