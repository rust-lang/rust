enum VecOrMap{
    vec: Vec<usize>,
    //~^ ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found `:`
    //~| HELP: enum variants can be `Variant`, `Variant = <integer>`, `Variant(Type, ..., TypeN)` or `Variant { fields: Types }`
    map: HashMap<String,usize>
}

fn main() {}
