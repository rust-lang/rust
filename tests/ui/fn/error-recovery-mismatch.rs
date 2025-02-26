// Used to ICE due to a size mismatch between the actual fake signature of `fold` and the
// generated signature used reporting the parameter mismatch at the call site.
// See issue #135124

trait A  {
    fn y(&self)
    {
        fn call() -> impl Sized {}
        self.fold(call(), call());
    }
    fn fold<T>(&self, _: T, &self._) {}
    //~^ ERROR unexpected `self` parameter in function
    //~| ERROR expected one of `)` or `,`, found `.`
    //~| ERROR identifier `self` is bound more than once in this parameter list
    //~| WARNING anonymous parameters are deprecated
    //~| WARNING this is accepted in the current edition
    //~| ERROR the placeholder `_` is not allowed within types
}

fn main() {}
