//@compile-flags: --test
#![warn(clippy::panic)]
use std::panic::panic_any;

fn main() {
    enum Enam {
        A,
    }
    let a = Enam::A;
    match a {
        Enam::A => {},
        _ => panic!(""),
        //~^ panic
    }
}

fn issue_13292() {
    panic_any("should lint")
    //~^ panic
}

#[test]
fn lonely_test() {
    enum Enam {
        A,
    }
    let a = Enam::A;
    match a {
        Enam::A => {},
        _ => panic!(""),
    }
}

#[cfg(test)]
mod tests {
    // should not lint in `#[cfg(test)]` modules
    #[test]
    fn test_fn() {
        enum Enam {
            A,
        }
        let a = Enam::A;
        match a {
            Enam::A => {},
            _ => panic!(""),
        }

        bar();
    }

    fn bar() {
        enum Enam {
            A,
        }
        let a = Enam::A;
        match a {
            Enam::A => {},
            _ => panic!(""),
        }
    }
}
