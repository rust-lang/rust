// Our diagnostics should be able to point to a specific input that caused an invalid
// identifier.

#![feature(macro_metavar_expr_concat)]

// See what we can do without expanding anything
macro_rules! pre_expansion {
    ($a:ident) => {
        ${concat("hi", " bye ")};
        ${concat("hi", "-", "bye")};
        ${concat($a, "-")};
    }
}

macro_rules! post_expansion {
    ($a:literal) => {
        const _: () = ${concat("hi", $a, "bye")};
        //~^ ERROR is not generating a valid identifier
        //~| NOTE this `${concat(..)}` invocation generated `hi!bye`, but '!' is not XID_Continue
        //~| NOTE see <https://doc.rust-lang.org/reference/identifiers.html> for the definition of valid identifiers
    }
}

post_expansion!("!");
//~^ NOTE in this expansion of post_expansion!
//~| NOTE in this expansion of post_expansion!
//~| NOTE in this expansion of post_expansion!

macro_rules! post_expansion_many {
    ($a:ident, $b:ident, $c:ident, $d:literal, $e:ident) => {
        const _: () = ${concat($a, $b, $c, $d, $e)};
        //~^ ERROR is not generating a valid identifier
        //~| NOTE this `${concat(..)}` invocation generated `abc.de`, but '.' is not XID_Continue
        //~| NOTE see <https://doc.rust-lang.org/reference/identifiers.html> for the definition of valid identifiers
    }
}

post_expansion_many!(a, b, c, ".d", e);
//~^ NOTE in this expansion of post_expansion_many!
//~| NOTE in this expansion of post_expansion_many!
//~| NOTE in this expansion of post_expansion_many!

fn main() {}
