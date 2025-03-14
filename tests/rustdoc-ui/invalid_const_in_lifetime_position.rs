trait X {
    type Y<'a>;
}
fn f<'a>(arg : Box<dyn X<Y<1> = &'a ()>>) {}
//~^ ERROR associated type takes 1 lifetime argument but 0 lifetime arguments
//~| ERROR associated type takes 0 generic arguments but 1 generic argument
//~| ERROR associated type takes 1 lifetime argument but 0 lifetime arguments
//~| ERROR associated type takes 0 generic arguments but 1 generic argument
//~| ERROR associated type takes 1 lifetime argument but 0 lifetime arguments
//~| ERROR associated type takes 0 generic arguments but 1 generic argument
//~| ERROR trait `X` is not dyn compatible
