//@ edition: 2021

fn main() {
    let foo = Some(123);
    match foo {
        Some(_)
            if let _ = (if let Some(0) = None
            //~^ ERROR: let chains are only allowed in Rust 2024 or later
                && true
            {
            } else {
            }) && let Some(3u32) = None => {}
        _ => {}
    }
}
