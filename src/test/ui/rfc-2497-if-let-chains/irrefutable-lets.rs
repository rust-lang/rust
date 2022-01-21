// check-pass

#![feature(if_let_guard, let_chains)]

use std::ops::Range;

fn main() {
    let opt = Some(None..Some(1));

    if let first = &opt && let Some(ref second) = first && let None = second.start {
    }
    if let Some(ref first) = opt && let second = first && let _third = second {
    }
    if let Some(ref first) = opt
        && let Range { start: local_start, end: _ } = first
        && let None = local_start {
    }

    match opt {
        Some(ref first) if let second = first && let _third = second => {},
        _ => {}
    }
    match opt {
        Some(ref first) if let Range { start: local_start, end: _ } = first
            && let None = local_start => {},
        _ => {}
    }

    while let first = &opt && let Some(ref second) = first && let None = second.start {
    }
    while let Some(ref first) = opt && let second = first && let _third = second {
    }
    while let Some(ref first) = opt
        && let Range { start: local_start, end: _ } = first
        && let None = local_start {
    }
}
