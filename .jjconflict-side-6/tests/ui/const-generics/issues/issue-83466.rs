// regression test for #83466- tests that generic arg mismatch errors between
// consts and types are not suppressed when there are explicit late bound lifetimes

struct S;
impl S {
    fn func<'a, U>(self) -> U {
        todo!()
    }
}
fn dont_crash<'a, U>() {
    S.func::<'a, 10_u32>()
    //~^ WARNING cannot specify lifetime arguments explicitly if late bound lifetime parameters are present
    //~^^ WARNING this was previously accepted by
    //~^^^ ERROR constant provided when a type was expected [E0747]
}

fn main() {}
