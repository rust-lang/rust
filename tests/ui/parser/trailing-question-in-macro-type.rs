macro_rules! fn_expr {
    ($return_type:ty : $body:expr) => {
        (|| -> $return_type { $body })()
    };
    ($body:expr) => {
        (|| $body)()
    };
}


fn main() {
    fn_expr!{ o?.when(|&i| i > 0)?.when(|&i| i%2 == 0) };
    //~^ ERROR cannot find value `o` in this scope
}
