use prelude::*; //~ ERROR unresolved import `prelude`

mod unresolved_env {
    use env; //~ ERROR unresolved import `env`

    include!(concat!(env!("NON_EXISTENT"), "/data.rs"));
}

mod nonexistent_env {
    include!(concat!(env!("NON_EXISTENT"), "/data.rs"));
    //~^ ERROR environment variable `NON_EXISTENT` not defined
}

mod erroneous_literal {
    include!(concat!("NON_EXISTENT"suffix, "/data.rs"));
    //~^ ERROR suffixes on string literals are invalid
}

fn main() {}
