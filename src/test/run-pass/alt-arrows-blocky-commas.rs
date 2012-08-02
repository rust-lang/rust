// no-reformat
// Testing the presense or absense of commas separating block-structure
// alt arm expressions

fn fun(_f: fn()) {
}

fn it(_f: fn() -> bool) {
}

fn main() {

    alt 0 {
      00 => {
      }
      01 => if true {
      } else {
      }
      03 => alt 0 {
        _ => ()
      }
      04 => do fun {
      }
      05 => for it {
      }
      06 => while false {
      }
      07 => loop {
      }
      08 => unsafe {
      }
      09 => unchecked {
      }
      10 => {
      },
      11 => if true {
      } else {
      },
      13 => alt 0 {
        _ => ()
      },
      14 => do fun {
      },
      15 => for it {
      },
      16 => while false {
      },
      17 => loop {
      },
      18 => unsafe {
      },
      19 => unchecked {
      },
      _ => ()
    }
}