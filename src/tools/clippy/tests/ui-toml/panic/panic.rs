//@compile-flags: --test
#![warn(clippy::panic)]

fn main() {
    enum Enam {
        A,
    }
    let a = Enam::A;
    match a {
        Enam::A => {},
        _ => panic!(""),
    }
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
