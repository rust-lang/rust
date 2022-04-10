trait BufferMut {}
struct Ctx<D>(D);

trait BufferUdpStateContext<B> {}
impl<B: BufferMut, C> BufferUdpStateContext<B> for C {}

trait StackContext
where
    Ctx<()>: for<'a> BufferUdpStateContext<&'a ()>,
    //~^ WARN where-clause bound is impossible to satisfy
{
    type Dispatcher;
}

trait TimerContext {
    type Handler;
}
impl<C> TimerContext for C
where
    C: StackContext,
    //~^ ERROR: is not satisfied [E0277]
{
    type Handler = Ctx<C::Dispatcher>;
    //~^ ERROR: is not satisfied [E0277]
}

struct EthernetWorker<C>(C)
where
    Ctx<()>: for<'a> BufferUdpStateContext<&'a ()>;
    //~^ WARN where-clause bound is impossible to satisfy

impl<C> EthernetWorker<C> {}
//~^ ERROR: is not satisfied [E0277]

fn main() {}
