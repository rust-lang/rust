// check-pass

macro_rules! foo { () => {
    let x = 1;
    macro_rules! bar { () => {x} }
    let _ = bar!();
}}

macro_rules! m { // test issue #31856
    ($n:ident) => (
        let a = 1;
        let $n = a;
    )
}

macro_rules! baz {
    ($i:ident) => {
        let mut $i = 2;
        $i = $i + 1;
    }
}

fn main() {
    foo! {};
    bar! {};

    let mut a = true;
    baz!(a);
}
