const COUNT: usize = 2;
struct Thing;
trait Dummy {}

const _: () = {
    impl Dummy for Thing where [i32; COUNT]: Sized {}
};

fn main() {}
