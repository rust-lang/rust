enum Bird {
    pub Duck,
    //~^ ERROR unnecessary visibility qualifier
    Goose,
    pub(crate) Dove
    //~^ ERROR unnecessary visibility qualifier
}


fn main() {
    let y = Bird::Goose;
}
