resource r(b: @mut int) {
    *b += 1;
}

fn main() {
    let b = @mut 0;
    {
        let p = some(r(b));
    }

    assert *b == 1;
}