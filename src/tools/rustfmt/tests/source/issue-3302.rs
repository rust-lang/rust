// rustfmt-style_edition: 2024

macro_rules! moo1 {
    () => {
        bar! {
"
"
        }
    };
}

macro_rules! moo2 {
    () => {
        bar! {
        "
"
        }
    };
}

macro_rules! moo3 {
    () => {
        42
        /*
        bar! {
        "
        toto
tata"
        }
        */
    };
}

macro_rules! moo4 {
    () => {
        bar! {
"
    foo
        bar
baz"
        }
    };
}
