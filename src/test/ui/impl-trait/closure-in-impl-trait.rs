// run-pass
// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

#![allow(unused_must_use)]
fn bug<T>() -> impl Iterator<Item = [(); { |x: u32| { x }; 4 }]> {
    std::iter::empty()
}

fn bug2<T>() -> impl Iterator<Item = [(); { |x: u32| unsafe { x }; 4 }]> {
    //~^ WARNING unnecessary `unsafe` block
    Box::new(std::iter::empty())
}

fn bug3<T>() -> impl Iterator<Item =
    [(); {|x: u32| unsafe { std::mem::transmute::<u32, i32>(x) }; 4 }]
> {
    Box::new(std::iter::empty())
}

fn ok<T>() -> Box<dyn Iterator<Item = [(); { |x: u32| { x }; 4 }]>> {
    Box::new(std::iter::empty())
}

fn ok2<T>() -> Box<dyn Iterator<Item = [(); { |x: u32| unsafe { x }; 4 }]>> {
    //~^ WARNING unnecessary `unsafe` block
    Box::new(std::iter::empty())
}

fn ok3<T>() -> Box<dyn Iterator<Item =
    [(); {|x: u32| unsafe { std::mem::transmute::<u32, i32>(x) }; 4 }]
>> {
    Box::new(std::iter::empty())
}

fn main() {
    for _item in ok::<u32>() {}
    for _item in ok2::<u32>() {}
    for _item in ok3::<u32>() {}
    for _item in bug::<u32>() {}
    for _item in bug2::<u32>() {}
    for _item in bug3::<u32>() {}
}
