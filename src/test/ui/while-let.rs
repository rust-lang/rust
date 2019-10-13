// run-pass

#[allow(dead_code)]
fn macros() {
    macro_rules! foo{
        ($p:pat, $e:expr, $b:block) => {{
            while let $p = $e $b
        }}
    }
    macro_rules! bar{
        ($p:pat, $e:expr, $b:block) => {{
            foo!($p, $e, $b)
        }}
    }

    foo!(_a, 1, { //~ WARN irrefutable while-let
        println!("irrefutable pattern");
    });
    bar!(_a, 1, { //~ WARN irrefutable while-let
        println!("irrefutable pattern");
    });
}

pub fn main() {
    while let _a = 1 { //~ WARN irrefutable while-let
        println!("irrefutable pattern");
        break;
    }
}
