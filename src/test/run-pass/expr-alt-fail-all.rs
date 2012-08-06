


// When all branches of an match expression result in fail, the entire
// match expression results in fail.
fn main() {
    let x =
        match true {
          true => { 10 }
          false => { match true { true => { fail } false => { fail } } }
        };
}
