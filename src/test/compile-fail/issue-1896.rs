type t<T> = { f: fn() -> T };

fn f<T>(_x: t<T>) {}

fn main() {
    let x: t<()> = { f: { || () } }; //~ ERROR expected & closure, found @ closure
    //~^ ERROR in field `f`, expected & closure, found @ closure
    f(x);
}