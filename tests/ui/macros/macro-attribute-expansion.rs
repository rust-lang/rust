//@ run-pass
macro_rules! descriptions {
    ($name:ident is $desc:expr) => {
        // Check that we will correctly expand attributes
        #[doc = $desc]
        #[allow(dead_code)]
        const $name : &'static str = $desc;
    }
}

// item
descriptions! { DOG is "an animal" }
descriptions! { RUST is "a language" }

pub fn main() {
}
