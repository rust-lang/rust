trait Tr {
    type Assoc;
}

struct W<T>(T);

impl Tr for W<i32> {
    type Assoc = u32;
}

impl Tr for W<u32> {
    type Assoc = i32;
}

fn needs_unit<T: Tr<Assoc = ()>>() {}

fn main() {
    needs_unit::<W<i32>>();
    //~^ ERROR type mismatch resolving `<W<i32> as Tr>::Assoc == ()`
}
