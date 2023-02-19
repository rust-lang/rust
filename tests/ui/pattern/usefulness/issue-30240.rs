fn main() {
    match "world" { //~ ERROR match is non-exhaustive
        "hello" => {}
    }

    match "world" { //~ ERROR match is non-exhaustive
        ref _x if false => {}
        "hello" => {}
    }
}
