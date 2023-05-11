fn main() {
    match "world" { //~ ERROR non-exhaustive patterns: `&_`
        "hello" => {}
    }

    match "world" { //~ ERROR non-exhaustive patterns: `&_`
        ref _x if false => {}
        "hello" => {}
    }
}
