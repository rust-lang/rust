struct Foo<'a, 'b: 'a>(&'a &'b ());

impl<'a, 'b> Foo<'a, 'b> {
    fn xmute(a: &'b ()) -> &'a () {
        unreachable!()
    }
}

pub fn foo<'a, 'b>(u: &'b ()) -> &'a () {
    Foo::<'a, 'b>::xmute(u)
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
