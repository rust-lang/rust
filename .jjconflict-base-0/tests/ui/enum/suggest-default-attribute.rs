pub enum Test { //~ HELP consider adding a derive
    #[default]
    //~^ ERROR cannot find attribute `default` in this scope
    First,
    Second,
}

fn main() {}
