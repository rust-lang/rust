pub enum Test { //~ HELP consider adding a derive
    #[default]
    //~^ ERROR cannot find attribute `default`
    First,
    Second,
}

fn main() {}
