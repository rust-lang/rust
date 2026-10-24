trait TraitNotAStruct {}

fn main() {
    TraitNotAStruct{ value: 0 };
    //~^ ERROR expected struct, variant or union type, found trait `TraitNotAStruct`
}
