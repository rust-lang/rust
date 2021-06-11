// revisions: full min
//[full] check-pass

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct Const<const P: &'static ()>;
//[min]~^ ERROR `&'static ()` is forbidden as the type of a const generic parameter

fn main() {
    const A: &'static () = unsafe {
        std::mem::transmute(10 as *const ())
    };

    let _ = Const::<{A}>;
}
