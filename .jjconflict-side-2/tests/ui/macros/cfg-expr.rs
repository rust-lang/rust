//@ run-pass
macro_rules! foo {
    ($e:expr, $n:ident) => {
        #[cfg($e)]
        macro_rules! $n {
            () => {}
        }

        #[cfg_attr($e, allow(non_snake_case))]
        #[cfg($e)]
        fn $n() {
            #[cfg($e)]
            $n!();
        }

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
foo!(false, haha);

fn main() {
    BAR();
    BAR!();
    baz();
    baz!();
    #[cfg(target_pointer_width = "64")]
    quux();
    #[cfg(target_pointer_width = "64")]
    quux!();
    #[cfg(panic = "unwind")]
    {
        let result = std::panic::catch_unwind(|| {
            haha();
        });
        assert!(result.is_err());
    }
}
