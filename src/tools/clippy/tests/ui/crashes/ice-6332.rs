fn cmark_check() {
    let mut link_err = false;
    macro_rules! cmark_error {
        ($bad:expr) => {
            *$bad = true;
        };
    }
    cmark_error!(&mut link_err);
}

pub fn main() {}
