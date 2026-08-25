// Ensure that we cannot move out of a fixed-size array (especially
// when the element type has a destructor).


struct D { _x: u8 }

impl Drop for D { fn drop(&mut self) { } }

fn main() {
    fn d() -> D { D { _x: 0 } }

    let _d1 = foo([d(), d(), d(), d()], 1);
    let _d3 = foo([d(), d(), d(), d()], 3);
}

fn foo(a: [D; 4], i: usize) -> D {
    a[i] //~ ERROR cannot move out of type `[D; 4]`, a non-copy array
}
