#[macro_export]
macro_rules! eq {
    (assert $a:expr, $b:expr) => {
        match (&$a, &$b) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    panic!(
                        "assertion failed: `(left == right)`\n  left: `{:?}`,\n right: `{:?}`",
                        left_val, right_val
                    );
                }
            }
        }
    };
}
