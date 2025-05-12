// #2554
// Do not add the beginning vert to the first match arm's pattern.

fn main() {
    match foo(|_| {
        bar(|_| {
            //
        })
    }) {
        Ok(()) => (),
        Err(_) => (),
    }
}
