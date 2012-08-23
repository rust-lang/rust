// error-pattern:squirrelcupcake
fn cmp() -> int {
    match (option::some('a'), option::none::<char>) {
        (option::some(_), _) => { fail ~"squirrelcupcake"; }
        (_, option::some(_)) => { fail; }
        _                    => { fail ~"wat"; }
    }
}

fn main() { log(error, cmp()); }
