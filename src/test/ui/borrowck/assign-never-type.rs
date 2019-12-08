// Regression test for issue 62165

// check-pass

pub fn main() {
    loop {
        match None {
            None => return,
            Some(val) => val,
        };
    };
}
