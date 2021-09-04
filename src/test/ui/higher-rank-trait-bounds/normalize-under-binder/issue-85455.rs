#![feature(unboxed_closures)]

trait SomeTrait<'a> {
    type Associated;
}

fn give_me_ice<T>() {
    callee::<fn(&()) -> <T as SomeTrait<'_>>::Associated>();
    //~^ ERROR: the trait bound `T: SomeTrait<'_>` is not satisfied
}

fn callee<T: Fn<(&'static (),)>>() {
    println!("{}", std::any::type_name::<<T as FnOnce<(&'static (),)>>::Output>());
}

fn main() {
    give_me_ice::<()>();
}
