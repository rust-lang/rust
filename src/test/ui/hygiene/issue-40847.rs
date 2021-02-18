// run-pass
macro_rules! gen {
    ($name:ident ( $($dol:tt $var:ident)* ) $($body:tt)*) => {
        macro_rules! $name {
            ($($dol $var:ident)*) => {
                $($body)*
            }
        }
    }
}

gen!(m($var) $var);

fn main() {
    let x = 1;
    assert_eq!(m!(x), 1);
}
