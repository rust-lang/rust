// error-pattern:fail


fn build1() -> Vec<isize> {
    vec![0, 0, 0, 0, 0, 0, 0]
}

fn build2() -> Vec<isize> {
    panic!();
}

struct Blk {
    node: Vec<isize>,
    span: Vec<isize>,
}

fn main() {
    let _blk = Blk {
        node: build1(),
        span: build2(),
    };
}
