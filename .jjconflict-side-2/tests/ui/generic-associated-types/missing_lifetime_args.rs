trait X {
    type Y<'a, 'b>;
}

struct Foo<'a, 'b, 'c> {
    a: &'a u32,
    b: &'b str,
    c: &'c str,
}

fn foo<'c, 'd>(_arg: Box<dyn X<Y = (&'c u32, &'d u32)>>) {}
//~^ ERROR missing generics for associated type

fn bar<'a, 'b, 'c>(_arg: Foo<'a, 'b>) {}
//~^ ERROR struct takes 3 lifetime arguments but 2 lifetime

fn f<'a>(_arg: Foo<'a>) {}
//~^ ERROR struct takes 3 lifetime arguments but 1 lifetime

fn main() {}
