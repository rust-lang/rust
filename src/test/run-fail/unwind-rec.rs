// error-pattern:fail

fn build() -> [int] {
    fail;
}

fn main() {
    let blk = {
        node: build()
    };
}