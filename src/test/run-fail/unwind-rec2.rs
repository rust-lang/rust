// error-pattern:fail

fn build1() -> [int] {
    [0,0,0,0,0,0,0]
}

fn build2() -> [int] {
    fail;
}

fn main() {
    let blk = {
        node: build1(),
        span: build2()
    };
}