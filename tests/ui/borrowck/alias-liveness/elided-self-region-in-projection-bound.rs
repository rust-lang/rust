//@ edition: 2024
//@ known-bug: unknown

// From crater regression in #161154 in `ty_python_semantic`.

#![forbid(unsafe_code)]

struct Args(Vec<u32>);

struct Checker<'a> {
    args: &'a Args,
    count: usize,
}

impl<'a> Checker<'a> {
    // The elided lifetime borrows from `&self`
    fn enumerate(&self) -> impl Iterator<Item = (usize, &Args)> + 'a {
        let args = self.args;
        args.0.iter().enumerate().map(move |(i, _)| (i, args))
    }

    // Using `'a` fixes it
    fn enumerate_named(&self) -> impl Iterator<Item = (usize, &'a Args)> + 'a {
        let args = self.args;
        args.0.iter().enumerate().map(move |(i, _)| (i, args))
    }

    fn bump(&mut self) {
        self.count += 1;
    }
}

fn elided() {
    let args = Args(vec![1, 2, 3]);
    let mut checker = Checker { args: &args, count: 0 };
    for (_i, _a) in checker.enumerate() {
        checker.bump();
    }
}

fn named() {
    let args = Args(vec![1, 2, 3]);
    let mut checker = Checker { args: &args, count: 0 };
    for (_i, _a) in checker.enumerate_named() {
        checker.bump();
    }
}

fn main() {
    elided();
    named();
}
