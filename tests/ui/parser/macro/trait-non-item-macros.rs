macro_rules! bah {
    ($a:expr) => {
        $a
    }; //~^ ERROR macro expansion ignores `expr` metavariable and any tokens following
}

trait Bar {
    bah!(2);
}

fn main() {
    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
