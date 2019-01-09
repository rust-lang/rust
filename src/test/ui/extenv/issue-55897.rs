use prelude::*; //~ ERROR unresolved import `prelude`

mod unresolved_env {
    use env;

    include!(concat!(env!("NON_EXISTENT"), "/data.rs"));
    //~^ ERROR cannot determine resolution for the macro `env`
}

mod nonexistent_env {
    include!(concat!(env!("NON_EXISTENT"), "/data.rs"));
    //~^ ERROR environment variable `NON_EXISTENT` not defined
}

fn main() {}
