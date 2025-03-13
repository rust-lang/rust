#![allow(unused)]
#![warn(clippy::match_as_ref)]

fn match_as_ref() {
    let owned: Option<()> = None;
    let borrowed: Option<&()> = match owned {
        //~^ match_as_ref
        None => None,
        Some(ref v) => Some(v),
    };

    let mut mut_owned: Option<()> = None;
    let borrow_mut: Option<&mut ()> = match mut_owned {
        //~^ match_as_ref
        None => None,
        Some(ref mut v) => Some(v),
    };
}

mod issue4437 {
    use std::error::Error;
    use std::fmt;
    use std::num::ParseIntError;

    #[derive(Debug)]
    struct E {
        source: Option<ParseIntError>,
    }

    impl Error for E {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            match self.source {
                //~^ match_as_ref
                Some(ref s) => Some(s),
                None => None,
            }
        }
    }

    impl fmt::Display for E {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            unimplemented!()
        }
    }
}

fn main() {
    // Don't lint
    let _ = match Some(0) {
        #[cfg(feature = "foo")]
        Some(ref x) if *x > 50 => None,
        Some(ref x) => Some(x),
        None => None,
    };
}
