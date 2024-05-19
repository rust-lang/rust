//@ known-bug: #93237
trait Trait {
    type Assoc;
}
impl Trait for () {
    type Assoc = ();
}

macro_rules! m {
    ([#$($t:tt)*] [$($open:tt)*] [$($close:tt)*]) => {
        m!{[$($t)*][$($open)*$($open)*][$($close)*$($close)*]}
    };
    ([] [$($open:tt)*] [$($close:tt)*]) => {
        fn _f() -> $($open)*()$($close)* {}
    };
}

m! {[###########][impl Trait<Assoc =][>]}
