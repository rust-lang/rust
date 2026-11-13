//@ run-pass

macro_rules! format_mbe {
    ($tt:tt) => {
        {
            #[allow(unused_variables)]
            let a = 123;
            format!($tt)
        }
    };
}

fn main() {
    let a = 0;
    assert_eq!(format_mbe!("{a}"), "0");
}
