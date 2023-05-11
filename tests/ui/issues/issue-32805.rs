// run-pass
fn const_mir() -> f32 { 9007199791611905.0 }

fn main() {
    let original = "9007199791611905.0"; // (1<<53)+(1<<29)+1
    let expected = "9007200000000000";

    assert_eq!(const_mir().to_string(), expected);
    assert_eq!(original.parse::<f32>().unwrap().to_string(), expected);
}
