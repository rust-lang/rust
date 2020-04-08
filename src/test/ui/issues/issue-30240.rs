fn main() {
    match "world" { //~ ERROR non-exhaustive patterns: `&Str(_)`
        "hello" => {}
    }

    match "world" { //~ ERROR non-exhaustive patterns: `&Str(_)`
        ref _x if false => {}
        "hello" => {}
    }
}
