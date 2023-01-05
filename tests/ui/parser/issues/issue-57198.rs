mod m {
    pub fn r#for() {}
}

fn main() {
    m::for();
    //~^ ERROR expected identifier, found keyword `for`
}
