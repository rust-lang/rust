//@ run-pass

#![deny(unreachable_patterns)]

fn main() {
    match (3, 42) {
        (a, _) | (_, a) if a > 10 => {}
        _ => unreachable!(),
    }

    match Some((3, 42)) {
        Some((a, _)) | Some((_, a)) if a > 10 => {}
        _ => unreachable!(),
    }

    match Some((3, 42)) {
        Some((a, _) | (_, a)) if a > 10 => {}
        _ => unreachable!(),
    }
}
