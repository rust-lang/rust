pub enum Test { //~ HELP consider adding `#[derive(Default)]` to this enum
    #[default]
    //~^ ERROR cannot find attribute `default` in this scope
    First,
    Second,
}

fn main() {}
