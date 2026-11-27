extern crate left;
extern crate right;

#[inline(never)]
fn generic<T>(value: u32) -> u32 {
    std::hint::black_box(value)
}

fn main() {
    std::hint::black_box(left::left_function(123_u32));
    std::hint::black_box(right::right_function(456_u32));

    std::hint::black_box(generic::<(left::Left, left::Left, &dyn Fn(left::Left))>(111_u32));
    std::hint::black_box(generic::<(right::Right, right::Right, &dyn Fn(right::Right))>(222_u32));
}
