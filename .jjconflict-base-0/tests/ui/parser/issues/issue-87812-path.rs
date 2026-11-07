macro_rules! foo {
    ( $f:path ) => {{
        let _: usize = $f; //~ERROR
    }};
}

struct Baz;

fn main() {
    foo!(Baz);
}
