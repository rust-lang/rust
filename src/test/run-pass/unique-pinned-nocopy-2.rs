resource r(i: @mutable int) {
    *i = *i + 1;
}

fn main() {
    let i = @mutable 0;
    {
        let j = ~r(i);
    }
    assert *i == 1;
}