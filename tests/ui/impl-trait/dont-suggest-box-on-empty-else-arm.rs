fn test() -> impl std::fmt::Debug {
    if true {
        "boo2"
    } else {
        //~^ ERROR `if` and `else` have incompatible types
    }
}

fn main() {}
