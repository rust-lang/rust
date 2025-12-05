//@ edition: 2021
//@ revisions: mismatch mismatch_async too_many too_few lt

#[cfg(mismatch)]
trait Uwu {
    fn owo(x: ()) -> impl Sized;
}

#[cfg(mismatch)]
impl Uwu for () {
    fn owo(_: u8) {}
    //[mismatch]~^ ERROR method `owo` has an incompatible type for trait
}

#[cfg(mismatch_async)]
trait AsyncUwu {
    async fn owo(x: ()) {}
}

#[cfg(mismatch_async)]
impl AsyncUwu for () {
    async fn owo(_: u8) {}
    //[mismatch_async]~^ ERROR method `owo` has an incompatible type for trait
}

#[cfg(too_many)]
trait TooMuch {
    fn calm_down_please() -> impl Sized;
}

#[cfg(too_many)]
impl TooMuch for () {
    fn calm_down_please(_: (), _: (), _: ()) {}
    //[too_many]~^ ERROR method `calm_down_please` has 3 parameters but the declaration in trait `TooMuch::calm_down_please` has 0
}

#[cfg(too_few)]
trait TooLittle {
    fn come_on_a_little_more_effort(_: (), _: (), _: ()) -> impl Sized;
}

#[cfg(too_few)]
impl TooLittle for () {
    fn come_on_a_little_more_effort() {}
    //[too_few]~^ ERROR method `come_on_a_little_more_effort` has 0 parameters but the declaration in trait `TooLittle::come_on_a_little_more_effort` has 3
}

#[cfg(lt)]
trait Lifetimes {
    fn early<'early, T>(x: &'early T) -> impl Sized;
}

#[cfg(lt)]
impl Lifetimes for () {
    fn early<'late, T>(_: &'late ()) {}
    //[lt]~^ ERROR method `early` has an incompatible type for trait
}

fn main() {}
