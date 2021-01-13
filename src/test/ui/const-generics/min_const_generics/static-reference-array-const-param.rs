fn a<const X: &'static [u32]>() {}
//~^ ERROR `&'static [u32]` is forbidden as the type of a const generic parameter

fn main() {
    a::<{&[]}>();
}
