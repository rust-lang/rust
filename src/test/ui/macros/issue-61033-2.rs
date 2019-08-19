// Regression test for issue #61033.

macro_rules! test2 {
    (
        $(* $id1:ident)*
        $(+ $id2:ident)*
    ) => {
        $( //~ERROR meta-variable `id1` repeats 2 times
            $id1 + $id2 // $id1 and $id2 may repeat different numbers of times
        )*
    }
}

fn main() {
    test2! {
        * a * b
        + a + b + c
    }
}
