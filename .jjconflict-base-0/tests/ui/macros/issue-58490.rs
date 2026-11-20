// Regression test for #58490

macro_rules! a {
    ( @1 $i:item ) => {
        a! { @2 $i }
    };
    ( @2 $i:item ) => {
        $i
    };
}
mod b {
    a! {
        @1
        #[macro_export]
        macro_rules! b { () => () }
    }
    #[macro_export]
    macro_rules! b { () => () }
    //~^ ERROR: the name `b` is defined multiple times
}
mod c {
    #[allow(unused_imports)]
    use crate::b;
}

fn main() {}
