enum Bird {
    pub Duck,
    //~^ ERROR visibility qualifiers are not permitted here
    Goose,
    pub(crate) Dove
    //~^ ERROR visibility qualifiers are not permitted here
}


fn main() {
    let y = Bird::Goose;
}
