resource r(i: @mut int) {
    *i = *i + 1;
}

fn main() {
    let i = @mut 0;
    {
        let j = ~r(i);
    }
    assert *i == 1;
}