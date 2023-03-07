// run-pass
#![feature(box_patterns)]

pub fn main() {
    match Box::new(100) {
      box x => {
        println!("{}", x);
        assert_eq!(x, 100);
      }
    }
}
