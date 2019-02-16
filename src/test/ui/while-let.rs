// run-pass

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

    foo!(a, 1, { //~ WARN irrefutable while-let
        println!("irrefutable pattern");
    });
    bar!(a, 1, { //~ WARN irrefutable while-let
        println!("irrefutable pattern");
    });
}

pub fn main() {
    while let a = 1 { //~ WARN irrefutable while-let
        println!("irrefutable pattern");
        break;
    }
}
