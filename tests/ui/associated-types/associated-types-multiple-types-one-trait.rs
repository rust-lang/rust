trait Foo {
    type X;
    type Y;
}

fn have_x_want_x<T:Foo<X=u32>>(t: &T)
{
    want_x(t);
}

fn have_x_want_y<T:Foo<X=u32>>(t: &T)
{
    want_y(t); //~ ERROR type mismatch
}

fn have_y_want_x<T:Foo<Y=i32>>(t: &T)
{
    want_x(t); //~ ERROR type mismatch
}

fn have_y_want_y<T:Foo<Y=i32>>(t: &T)
{
    want_y(t);
}

fn have_xy_want_x<T:Foo<X=u32,Y=i32>>(t: &T)
{
    want_x(t);
}

fn have_xy_want_y<T:Foo<X=u32,Y=i32>>(t: &T)
{
    want_y(t);
}

fn have_xy_want_xy<T:Foo<X=u32,Y=i32>>(t: &T)
{
    want_x(t);
    want_y(t);
}

fn want_x<T:Foo<X=u32>>(t: &T) { }

fn want_y<T:Foo<Y=i32>>(t: &T) { }

fn main() { }
