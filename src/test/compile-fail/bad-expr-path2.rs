// error-pattern: unresolved name: m1::a

mod m1 {
    #[legacy_exports];
    mod a {
        #[legacy_exports]; }
}

fn main(args: ~[str]) { log(debug, m1::a); }
