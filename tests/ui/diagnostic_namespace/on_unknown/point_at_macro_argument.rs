//! Test that the `label` span points at the identifier passed to the macro.

#![feature(diagnostic_on_unknown)]
#![crate_type = "lib"]

mod things {}

macro_rules! mac {
    ($thing: ident) => {{
        const _x: u32 = {
            #[diagnostic::on_unknown(label = "you did the bad thing")]
            use things::$thing;
            //~^ ERROR unresolved import `things::what` [E0432]
            //~| ERROR unresolved import `things::what2` [E0432]
            $thing
        };
    }};
}

fn stuff() {
    mac!(what);
    //~^ NOTE you did the bad thing
    //~| NOTE in this expansion of mac!

    mac!(//~ NOTE in this expansion of mac!

        what2
        //~^ NOTE you did the bad thing

    );
}
