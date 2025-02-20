const _A: f64 = 1em;
    //~^ ERROR invalid suffix `em` for number literal
const _B: f64 = 1e0m;
    //~^ ERROR invalid suffix `m` for float literal
const _C: f64 = 1e_______________0m;
    //~^ ERROR invalid suffix `m` for float literal
const _D: f64 = 1e_______________m;
    //~^ ERROR invalid suffix `e_______________m` for number literal

fn main() {}
