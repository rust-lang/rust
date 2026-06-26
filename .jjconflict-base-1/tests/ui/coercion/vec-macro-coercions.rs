//@ check-pass

fn main() {
    let functions = &vec![
        |x: i32| -> i32 { x + 3 },
        |x: i32| -> i32 { x + 3 },
    ];

    let string = String::new();
    let a = vec![&string, "abc"];
    let b = vec!["abc", &string];
}
