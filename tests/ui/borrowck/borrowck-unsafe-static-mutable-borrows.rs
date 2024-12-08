//@ run-pass

// Test file taken from issue 45129 (https://github.com/rust-lang/rust/issues/45129)

struct Foo {
    x: [usize; 2],
}

static mut SFOO: Foo = Foo { x: [23, 32] };

impl Foo {
    fn x(&mut self) -> &mut usize {
        &mut self.x[0]
    }
}

fn main() {
    unsafe {
        let sfoo: *mut Foo = &mut SFOO;
        //~^ WARN mutable reference to mutable static is discouraged [static_mut_refs]
        let x = (*sfoo).x();
        (*sfoo).x[1] += 1;
        *x += 1;
    }
}
