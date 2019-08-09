// build-pass (FIXME(62277): could be check-pass?)

fn from_stdin(min: u64) -> Vec<u64> {
    use std::io::BufRead;

    let stdin = std::io::stdin();
    let stdin = stdin.lock();

    stdin.lines()
        .map(Result::unwrap)
        .map(|val| val.parse())
        .map(Result::unwrap)
        .filter(|val| *val >= min)
        .collect()
}

fn main() {}
