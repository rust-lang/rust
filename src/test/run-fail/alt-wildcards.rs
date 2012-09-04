// error-pattern:squirrelcupcake
fn cmp() -> int {
    match (option::Some('a'), option::None::<char>) {
        (option::Some(_), _) => { fail ~"squirrelcupcake"; }
        (_, option::Some(_)) => { fail; }
        _                    => { fail ~"wat"; }
    }
}

fn main() { log(error, cmp()); }
