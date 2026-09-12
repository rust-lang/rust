//@aux-build: proc_macros.rs
#![warn(clippy::exit)]

fn not_main() {
    if true {
        std::process::exit(4);
        //~^ exit
    }
}

fn main() {
    if true {
        std::process::exit(2);
    };
    not_main();
    std::process::exit(1);
}

fn issue17082() {
    macro_rules! mac {
        ($x:expr) => {{
            $x;
        }};
        (ex $x:expr) => {{
            std::process::exit($x);
            //~^ exit
        }};
    }
    mac!(std::process::exit(1));
    //~^ exit
    mac!(ex 1);
    proc_macros::external! {
        std::process::exit(1);
    }
}
