// run-rustfix

#![warn(clippy::pats_with_wild_match_arm)]

fn main() {
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        "bar" | _ => {
            dbg!("matched (bar or) wild");
        },
    };
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        "bar" | "bar2" | _ => {
            dbg!("matched (bar or bar2 or) wild");
        },
    };
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        _ | "bar" | _ => {
            dbg!("matched (bar or) wild");
        },
    };
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        _ | "bar" => {
            dbg!("matched (bar or) wild");
        },
    };
}
