


// When all branches of an alt expression result in fail, the entire
// alt expression results in fail.
fn main() {
    let x =
        alt true {
          true { 10 }
          false { alt true { true { fail } false { fail } } }
        };
}
