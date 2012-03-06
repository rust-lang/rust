resource r(b: @mutable int) {
    *b += 1;
}

fn main() {
    let b = @mutable 0;
    {
        let p = some(r(b));
    }

    assert *b == 1;
}