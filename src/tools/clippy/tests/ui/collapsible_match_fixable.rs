#![warn(clippy::collapsible_match)]
#![allow(clippy::single_match, clippy::redundant_guards)]

fn issue16558() {
    let opt = Some(1);
    let _ = match opt {
        Some(s) => {
            if s == 1 { s } else { 1 }
            //~^ collapsible_match
        },
        _ => 1,
    };

    match opt {
        Some(s) => {
            (if s == 1 {
                //~^ collapsible_match
                todo!()
            })
        },
        _ => {},
    };

    let _ = match opt {
        Some(s) if s > 2 => {
            if s == 1 { s } else { 1 }
            //~^ collapsible_match
        },
        _ => 1,
    };
}
