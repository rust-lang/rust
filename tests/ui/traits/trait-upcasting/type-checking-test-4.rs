trait Foo<'a>: Bar<'a, 'a> {}
trait Bar<'a, 'b> {
    fn get_b(&self) -> Option<&'a u32> {
        None
    }
}

fn test_correct(x: &dyn Foo<'static>) {
    let _ = x as &dyn Bar<'static, 'static>;
}

fn test_correct2<'a>(x: &dyn Foo<'a>) {
    let _ = x as &dyn Bar<'_, '_>;
}

fn test_wrong1<'a>(x: &dyn Foo<'static>, y: &'a u32) {
    let _ = x as &dyn Bar<'static, 'a>; // Error
                                        //~^ ERROR lifetime may not live long enough
}

fn test_wrong2<'a>(x: &dyn Foo<'static>, y: &'a u32) {
    let _ = x as &dyn Bar<'a, 'static>; // Error
                                        //~^ ERROR lifetime may not live long enough
}

fn test_wrong3<'a>(x: &dyn Foo<'a>) -> Option<&'static u32> {
    let y = x as &dyn Bar<'_, '_>;
    y.get_b() // ERROR
              //~^ ERROR lifetime may not live long enough
}

fn test_wrong4<'a>(x: &dyn Foo<'a>) -> Option<&'static u32> {
    <_ as Bar>::get_b(x) // ERROR
                         //~^ ERROR lifetime may not live long enough
}

fn test_wrong5<'a>(x: &dyn Foo<'a>) -> Option<&'static u32> {
    <_ as Bar<'_, '_>>::get_b(x) // ERROR
                                 //~^ ERROR lifetime may not live long enough
}

fn test_wrong6<'a>(x: &dyn Foo<'a>) -> Option<&'static u32> {
    let y = x as &dyn Bar<'_, '_>;
    y.get_b(); // ERROR
    let z = y;
    z.get_b() // ERROR
              //~^ ERROR lifetime may not live long enough
}

fn main() {}
