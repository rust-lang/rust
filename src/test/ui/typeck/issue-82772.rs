// edition:2018

fn main() {
    use a::LocalModPrivateStruct;
    let Box { 1: _, .. }: Box<()>; //~ ERROR cannot match on
    let LocalModPrivateStruct { 1: _, .. } = LocalModPrivateStruct::default();
    //~^ ERROR cannot match on
}

mod a {
    #[derive(Default)]
    pub struct LocalModPrivateStruct(u8, u8);
}
