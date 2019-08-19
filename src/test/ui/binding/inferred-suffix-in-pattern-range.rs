// run-pass

pub fn main() {
    let x = 2;
    let x_message = match x {
      0 ..= 1    => { "not many".to_string() }
      _          => { "lots".to_string() }
    };
    assert_eq!(x_message, "lots".to_string());

    let y = 2;
    let y_message = match y {
      0 ..= 1    => { "not many".to_string() }
      _          => { "lots".to_string() }
    };
    assert_eq!(y_message, "lots".to_string());

    let z = 1u64;
    let z_message = match z {
      0 ..= 1    => { "not many".to_string() }
      _          => { "lots".to_string() }
    };
    assert_eq!(z_message, "not many".to_string());
}
