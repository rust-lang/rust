struct Message {
    field2: Vec<
      "MessageEntity"
    >,
    field3: Vec<
      1
    >,
    field4: Vec<
      2 ,    3
    >,

}

struct RectangularArray<T, const WIDTH: usize, const HEIGHT: usize> {
    array: [[T; WIDTH]; HEIGHT],
}

fn main() {
  const X: usize = 7;
  let x: RectangularArray<i32, 2, 4>;
  let y: RectangularArray<i32,  X,  {2 
  * 2} >;
}

fn foo<const X: usize>() {
    const Y: usize = X * 2;
    static Z: (usize, usize) = (X, X);

    struct Foo([i32; X]);
}

type Foo<const N: usize> = [i32; N + 1];

pub trait Foo: Bar<{Baz::COUNT}> {
	const ASD: usize;
}

// #4263
fn const_generics_on_params<
    // AAAA
        const BBBB: usize,
    /* CCCC */
    const DDDD: usize,
    >() {}
