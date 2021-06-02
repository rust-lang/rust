// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

fn nested1<T>() -> impl Iterator<Item =
    [(); {|x: u32| { fn x() {
        std::mem::transmute::<(), ()>(());
        //~^ ERROR call to unsafe function is unsafe
    } }; 4 }]
> {
    Box::new(std::iter::empty())
}

fn nested2<T>() -> Box<dyn Iterator<Item =
    [(); {|x: u32| { fn x() {
        std::mem::transmute::<(), ()>(());
        //~^ ERROR call to unsafe function is unsafe
    } }; 4 }]
>> {
    Box::new(std::iter::empty())
}

unsafe fn nested3<T>() -> impl Iterator<Item =
    [(); {|x: u32| { fn x() {
        std::mem::transmute::<(), ()>(());
        //~^ ERROR call to unsafe function is unsafe
    } }; 4 }]
> {
    Box::new(std::iter::empty())
}

unsafe fn nested4<T>() -> Box<dyn Iterator<Item =
    [(); {|x: u32| { fn x() {
        std::mem::transmute::<(), ()>(());
        //~^ ERROR call to unsafe function is unsafe
    } }; 4 }]
>> {
    Box::new(std::iter::empty())
}

fn main() {}
