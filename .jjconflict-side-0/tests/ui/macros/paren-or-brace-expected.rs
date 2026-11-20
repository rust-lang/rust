macro_rules! foo {
    ( $( $i:ident ),* ) => {
        $[count($i)]
        //~^ ERROR expected `(` or `{`, found `[`
        //~| ERROR
    };
}

fn main() {}
