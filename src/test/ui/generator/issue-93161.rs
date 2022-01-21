// edition:2021
// build-pass

#![feature(never_type)]

enum Never {}
fn never() -> Never {
    panic!()
}

async fn includes_never(crash: bool, x: u32) -> u32 {
    let mut result = async { x * x }.await;
    if !crash {
        return result;
    }
    #[allow(unused)]
    let bad = never();
    result *= async { x + x }.await;
    drop(bad);
    result
}

fn main() {
    let _ = includes_never(false, 4);
}
