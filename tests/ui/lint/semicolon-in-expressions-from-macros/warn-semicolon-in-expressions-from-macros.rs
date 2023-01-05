// check-pass
// Ensure that trailing semicolons cause warnings by default

macro_rules! foo {
    () => {
        true; //~  WARN trailing semicolon in macro
              //~| WARN this was previously
    }
}

fn main() {
    let _val = match true {
        true => false,
        _ => foo!()
    };
}
