// run-rustfix

#![warn(clippy::manual_map)]

fn main() {
    let _ = match Some(0) {
        Some(x) => Some({
            let y = (String::new(), String::new());
            (x, y.0)
        }),
        None => None,
    };
}
