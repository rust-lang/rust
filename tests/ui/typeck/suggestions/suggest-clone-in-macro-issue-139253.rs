#[derive(Debug, Clone)]
struct Struct { field: S }

#[derive(Debug, Clone)]
struct S;

macro_rules! expand {
    ($ident:ident) => { Struct { $ident } }
}

fn test1() {
    let field = &S;
    let a: Struct = dbg!(expand!(field)); //~ ERROR mismatched types [E0308]
    let b: Struct = dbg!(Struct { field }); //~ ERROR mismatched types [E0308]
    let c: S = dbg!(field); //~ ERROR mismatched types [E0308]
    let c: S = dbg!(dbg!(field)); //~ ERROR mismatched types [E0308]
}

fn main() {}
