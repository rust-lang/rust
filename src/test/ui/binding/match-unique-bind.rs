// run-pass
#![feature(box_patterns)]
#![feature(box_syntax)]

pub fn main() {
    match box 100 {
      box x => {
        println!("{}", x);
        assert_eq!(x, 100);
      }
    }
}
