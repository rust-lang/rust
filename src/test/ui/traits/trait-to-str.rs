// run-pass
#![allow(non_camel_case_types)]


trait to_str {
    fn to_string_(&self) -> String;
}

impl to_str for isize {
    fn to_string_(&self) -> String { self.to_string() }
}

impl<T:to_str> to_str for Vec<T> {
    fn to_string_(&self) -> String {
        format!("[{}]",
                self.iter()
                    .map(|e| e.to_string_())
                    .collect::<Vec<String>>()
                    .join(", "))
    }
}

pub fn main() {
    assert_eq!(1.to_string_(), "1".to_string());
    assert_eq!((vec![2, 3, 4]).to_string_(), "[2, 3, 4]".to_string());

    fn indirect<T:to_str>(x: T) -> String {
        format!("{}!", x.to_string_())
    }
    assert_eq!(indirect(vec![10, 20]), "[10, 20]!".to_string());

    fn indirect2<T:to_str>(x: T) -> String {
        indirect(x)
    }
    assert_eq!(indirect2(vec![1]), "[1]!".to_string());
}
