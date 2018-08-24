fn main() {
    match &Some(3) {
        &None => 1
        &Some(2) => { 3 }
        _ => 2
    };
}
