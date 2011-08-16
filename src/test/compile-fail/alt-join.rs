// error-pattern:unresolved name
// a good test that we merge paths correctly in the presence of a
// variable that's used before it's declared

fn my_fail() -> ! { fail; }

fn main() {
    alt (true) {
      false { my_fail(); }
      true {}
    }

    log x;
    let x:int;
}