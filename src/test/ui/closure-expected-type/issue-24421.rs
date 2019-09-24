// build-pass (FIXME(62277): could be check-pass?)

fn test<F: Fn(&u64, &u64)>(f: F) {}

fn main() {
    test(|x,      y     | {});
    test(|x:&u64, y:&u64| {});
    test(|x:&u64, y     | {});
    test(|x,      y:&u64| {});
}
