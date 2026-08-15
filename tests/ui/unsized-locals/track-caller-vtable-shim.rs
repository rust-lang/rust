//@ run-pass

#![feature(unsized_fn_params)]
#![allow(internal_features)]

trait TrackedByValue {
    #[track_caller]
    fn consume(self, expected_line: u32);
}

impl TrackedByValue for u8 {
    fn consume(self, expected_line: u32) {
        assert_eq!(self, 7);
        assert_eq!(std::panic::Location::caller().line(), expected_line);
    }
}

fn main() {
    let obj = Box::new(7_u8) as Box<dyn TrackedByValue>;
    obj.consume(line!());
}
