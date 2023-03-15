use std::time::Instant;

fn main() {
    let begin = Instant::now();
    for _ in 0..100_000 {}
    let time = begin.elapsed();
    println!("The loop took around {}s", time.as_secs());
}
