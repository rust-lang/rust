// Make sure that destructors get run on slice literals
struct foo {
    x: @mut int,
}

impl foo : Drop {
    fn finalize(&self) {
        *self.x += 1;
    }
}

fn foo(x: @mut int) -> foo {
    foo {
        x: x
    }
}

fn main() {
    let x = @mut 0;
    {
        let l = &[foo(x)];
        assert *l[0].x == 0;
    }
    assert *x == 1;
}
