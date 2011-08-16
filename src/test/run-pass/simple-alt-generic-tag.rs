

tag opt<T> { none; }

fn main() {
    let x = none::<int>; alt x { none::<int>. { log "hello world"; } }
}
