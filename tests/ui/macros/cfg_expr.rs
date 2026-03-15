//@ run-pass

macro_rules! foo {
    ($e:expr, $n:ident) => {
        #[cfg_attr($e, allow(non_snake_case))]
        #[cfg($e)]
        fn $n() {}

        #[cfg_attr(not($e), allow(unused))]
        #[cfg(not($e))]
        fn $n() {
            panic!()
        }
    }
}

foo!(true, BAR);
foo!(any(true, unix, target_pointer_width = "64"), baz);
foo!(target_pointer_width = "64", quux);

fn main() {
    BAR();
    baz();
    #[cfg(target_pointer_width = "64")]
    quux();
}
