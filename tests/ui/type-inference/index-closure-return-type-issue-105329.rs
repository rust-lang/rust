//@ edition:2024
// Regression test for https://github.com/rust-lang/rust/issues/105329
//
// When the index type comes from an ambiguous parsing call in a closure,
// report the error at the parsing call and label the index expression.
// Previously, the error only highlighted the whole indexing expression.

fn known_index_unknown_element() {
    let rows = vec![Vec::new()];
    //~^ ERROR type annotations needed
    rows[0usize].clone();
}

fn known_index_unknown_receiver() {
    let values = Vec::new();
    //~^ ERROR type annotations needed
    let _ = values[0usize].len();
}

fn explicit_index_type() {
    let mut rows: Vec<Vec<usize>> = vec![vec![]; 10];
    let f = |text: &str| text.parse::<usize>().unwrap();
    rows[f("2")].push(10);
}

fn main() {
    let s = "2";
    let mut vec: Vec<Vec<usize>> = vec![vec![]; 10];
    let f = |x: &str| x.parse().unwrap();
    vec[f(s)].push(10);
    //~^^ ERROR type annotations needed
}
