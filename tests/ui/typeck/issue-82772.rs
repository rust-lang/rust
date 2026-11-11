//@ edition:2018

fn main() {
    use a::ModPrivateStruct;
    let Box { 0: _, .. }: Box<()>; //~ ERROR field `0` of
    //~^ ERROR type `boxed::BoxRaw<()>` is private

    let Box { 1: _, .. }: Box<()>; //~ ERROR field `1` of
    let ModPrivateStruct { 1: _, .. } = ModPrivateStruct::default(); //~ ERROR field `1` of
}

mod a {
    #[derive(Default)]
    pub struct ModPrivateStruct(u8, u8);
}
