fn main() {
    let x: Vec<i32> = vec![1, 2, 3].into_iter().collect::<Vec<i32>>().sort_by_key(|i| i); //~ ERROR mismatched types
    vec![1, 2, 3].into_iter().collect::<Vec<i32>>().sort_by_key(|i| i).sort(); //~ ERROR no method named `sort` found for unit type `()` in the current scope
}

fn foo(mut s: String) -> String {
    s.push_str("asdf") //~ ERROR mismatched types
}
