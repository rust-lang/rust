// Ensure that trailing semicolons cause errors by default

macro_rules! foo {
    () => {
        true; //~  ERROR trailing semicolon in macro
              //~| WARN this was previously
    }
}

fn main() {
    let _val = match true {
        true => false,
        _ => foo!()
    };
}
