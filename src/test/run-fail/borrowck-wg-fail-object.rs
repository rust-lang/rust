#[feature(managed_boxes)];

// error-pattern:borrowed

trait Foo {
    fn foo(&self, @mut int);
}

impl Foo for int {
    fn foo(&self, x: @mut int) {
        *x += *self;
    }
}

fn main() {
    let x = @mut 3_i;
    let y = x as @mut Foo;

    // The call to `y.foo(...)` should freeze `y` (and thus also `x`,
    // since `x === y`). It is thus an error when `foo` tries to
    // mutate `x`.
    y.foo(x);
}
