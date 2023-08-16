// run-fail
//@error-in-other-file:squirrelcupcake
//@ignore-target-emscripten no processes

fn cmp() -> isize {
    match (Some('a'), None::<char>) {
        (Some(_), _) => {
            panic!("squirrelcupcake");
        }
        (_, Some(_)) => {
            panic!();
        }
        _ => {
            panic!("wat");
        }
    }
}

fn main() {
    println!("{}", cmp());
}
