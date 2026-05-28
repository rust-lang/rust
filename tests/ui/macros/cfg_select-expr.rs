//@ run-pass
#![allow(unreachable_cfg_select_predicates)]

macro_rules! foo {
    ($e:expr, $n:ident) => {
        cfg_select! {
            $e => {
                macro_rules! $n {
                    () => {}
                }
            }
            _ => {}
        }

        cfg_select! {
            $e => {
                #[cfg_attr($e, allow(non_snake_case))]
                fn $n() {
                    cfg_select! {
                        $e => {
                            $n!();
                        }
                        _ => {}
                    }
                }
            }
            not($e) => {
                #[cfg_attr(not($e), allow(unused))]
                fn $n() {
                    panic!()
                }
            }
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
