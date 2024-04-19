//@ known-bug: #124021
type Opaque2<'a> = impl Sized + 'a;

fn test2() -> impl for<'a, 'b> Fn((&'a str, &'b str)) -> (Opaque2<'a>, Opaque2<'a>) {
    |x| x
}
