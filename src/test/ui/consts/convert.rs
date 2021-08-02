// run-pass

#![feature(const_trait_impl)]
#![feature(const_identity_convert)]

fn main() {
    const fn from(x: i32) -> i32 {
        i32::from(x)
    }

    const FOO: i32 = from(42);
    assert_eq!(FOO, 42);

    const fn into(x: Vec<String>) -> Vec<String> {
        x.into()
    }

    const BAR: Vec<String> = into(Vec::new());
    assert_eq!(BAR, Vec::<String>::new());
}
