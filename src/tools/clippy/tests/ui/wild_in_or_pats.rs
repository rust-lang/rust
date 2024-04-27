#![warn(clippy::wildcard_in_or_patterns)]

fn main() {
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        "bar" | _ => {
            //~^ ERROR: wildcard pattern covers any other pattern as it will match anyway
            dbg!("matched (bar or) wild");
        },
    };
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        "bar" | "bar2" | _ => {
            //~^ ERROR: wildcard pattern covers any other pattern as it will match anyway
            dbg!("matched (bar or bar2 or) wild");
        },
    };
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        _ | "bar" | _ => {
            //~^ ERROR: wildcard pattern covers any other pattern as it will match anyway
            dbg!("matched (bar or) wild");
        },
    };
    match "foo" {
        "a" => {
            dbg!("matched a");
        },
        _ | "bar" => {
            //~^ ERROR: wildcard pattern covers any other pattern as it will match anyway
            dbg!("matched (bar or) wild");
        },
    };
}
