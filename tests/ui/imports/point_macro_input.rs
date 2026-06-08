#![crate_type = "lib"]

mod things { }

macro_rules! mac1 {
    ($thing: ident) => {{
        const _x: u32 = things::$thing;
        //~^ NOTE due to this macro variable
    }}
}

macro_rules! mac2 {
    ($thing: ident) => {{
        const _x: u32 = {
            use things::$thing;
            //~^ ERROR unresolved import `things::what2` [E0432]
            $thing
        };
    }}
}


fn foo(){
    mac1!(

        what1
        //~^ ERROR cannot find value `what1` in module `things` [E0425]
        //~| NOTE not found in `things`

    );
    mac2!(//~ NOTE in this expansion of mac2!

        what2
        //~^ NOTE no `what2` in `things`
    );
}
