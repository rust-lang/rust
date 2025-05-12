// Regression test for issue #115780.
// Ensure that we don't emit a parse error for the token sequence `Ident "<" Ty` in pattern position
// if we are inside a macro call since it can be valid input for a subsequent macro rule.
// See also #103534.

//@ check-pass

macro_rules! mdo {
    ($p: pat =<< $e: expr ; $( $t: tt )*) => {
        $e.and_then(|$p| mdo! { $( $t )* })
    };
    (ret<$ty: ty> $e: expr;) => { Some::<$ty>($e) };
}

fn main() {
    mdo! {
        x_val =<< Some(0);
        y_val =<< Some(1);
        ret<(i32, i32)> (x_val, y_val);
    };
}
