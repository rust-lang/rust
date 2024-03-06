//@ run-pass
trait ToRef<'a> {
    type Ref: 'a;
}

impl<'a, U: 'a> ToRef<'a> for U {
    type Ref = &'a U;
}

fn example<'a, T>(value: &'a T) -> (<T as ToRef<'a>>::Ref, u32) {
    (value, 0)
}

fn main() {
    example(&0);
}
