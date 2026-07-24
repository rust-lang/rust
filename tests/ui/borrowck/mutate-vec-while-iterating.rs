fn main() {
    let mut values = vec![1, 2, 3];

    for value in &values {
        if *value == 2 {
            values.push(4); //~ ERROR E0502
        }
    }
}
