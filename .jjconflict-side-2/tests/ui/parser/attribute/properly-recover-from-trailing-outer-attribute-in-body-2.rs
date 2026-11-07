// Issue #121647: recovery path leaving unemitted error behind

macro_rules! the_macro {
    ( $foo:stmt ; $bar:stmt ; ) => {
        #[cfg()]
        $foo //~ ERROR expected `;`, found `#`

        #[cfg(false)]
        $bar
    };
}

fn main() {
    the_macro!( (); (); );
}
