// xfail-test
enum PureCounter { PureCounter(uint) }

pure fn each(self: PureCounter, blk: fn(v: &uint)) {
    let PureCounter(ref x) = self;
    blk(x);
}

fn main() {}
