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

fn it_takes_two(_f: &Foo, _g: &mut Foo) {
}

fn main() {
    let x = @mut 3_i;
    let y = x as @mut Foo;
    let z = y;

    it_takes_two(y, z);
}
