// Make sure that destructors get run on slice literals
struct foo {
    let x: @mut int;
    new(x: @mut int) { self.x = x; }
    drop { *self.x += 1; }
}

fn main() {
    let x = @mut 0;
    {
        let l = &[foo(x)];
        assert *l[0].x == 0;
    }
    assert *x == 1;
}
