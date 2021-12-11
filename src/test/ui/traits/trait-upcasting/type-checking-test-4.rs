// ignore-compare-mode-nll
#![feature(trait_upcasting)]
#![allow(incomplete_features)]

trait Foo<'a>: Bar<'a, 'a> {}
trait Bar<'a, 'b> {
    fn get_b(&self) -> Option<&'a u32> {
        None
    }
}

fn test_correct(x: &dyn Foo<'static>) {
    let _ = x as &dyn Bar<'static, 'static>;
}

fn test_wrong1<'a>(x: &dyn Foo<'static>, y: &'a u32) {
    let _ = x as &dyn Bar<'static, 'a>; // Error
                                        //~^ ERROR mismatched types
}

fn test_wrong2<'a>(x: &dyn Foo<'static>, y: &'a u32) {
    let _ = x as &dyn Bar<'a, 'static>; // Error
                                        //~^ ERROR mismatched types
}

fn test_wrong3<'a>(x: &dyn Foo<'a>) -> Option<&'static u32> {
    let y = x as &dyn Bar<'_, '_>;
    //~^ ERROR `x` has lifetime `'a` but it needs to satisfy a `'static` lifetime requirement
    y.get_b() // ERROR
}

fn test_wrong4<'a>(x: &dyn Foo<'a>) -> Option<&'static u32> {
    <_ as Bar>::get_b(x) // ERROR
    //~^ ERROR `x` has lifetime `'a` but it needs to satisfy a `'static` lifetime requirement
}

fn test_wrong5<'a>(x: &dyn Foo<'a>) -> Option<&'static u32> {
    <_ as Bar<'_, '_>>::get_b(x) // ERROR
    //~^ ERROR `x` has lifetime `'a` but it needs to satisfy a `'static` lifetime requirement
}

fn test_wrong6<'a>(x: &dyn Foo<'a>) -> Option<&'static u32> {
    let y = x as &dyn Bar<'_, '_>;
    //~^ ERROR `x` has lifetime `'a` but it needs to satisfy a `'static` lifetime requirement
    y.get_b(); // ERROR
    let z = y;
    z.get_b() // ERROR
}

fn main() {}
