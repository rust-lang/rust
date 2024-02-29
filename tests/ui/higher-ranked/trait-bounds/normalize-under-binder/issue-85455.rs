#![feature(unboxed_closures)]

trait SomeTrait<'a> {
    type Associated;
}

fn give_me_ice<T>() {
    callee::<fn(&()) -> <T as SomeTrait<'_>>::Associated>();
    //~^ ERROR trait `SomeTrait<'_>` is not implemented for `T`
    //~| ERROR trait `SomeTrait<'_>` is not implemented for `T`
}

fn callee<T: Fn<(&'static (),)>>() {
    println!("{}", std::any::type_name::<<T as FnOnce<(&'static (),)>>::Output>());
}

fn main() {
    give_me_ice::<()>();
}
