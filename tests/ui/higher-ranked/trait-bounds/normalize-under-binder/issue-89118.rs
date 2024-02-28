trait BufferMut {}
struct Ctx<D>(D);

trait BufferUdpStateContext<B> {}
impl<B: BufferMut, C> BufferUdpStateContext<B> for C {}

trait StackContext
where
    Ctx<()>: for<'a> BufferUdpStateContext<&'a ()>,
{
    type Dispatcher;
}

trait TimerContext {
    type Handler;
}
impl<C> TimerContext for C
where
    C: StackContext,
    //~^ ERROR: the trait
{
    type Handler = Ctx<C::Dispatcher>;
    //~^ ERROR: the trait
}

struct EthernetWorker<C>(C)
where
    Ctx<()>: for<'a> BufferUdpStateContext<&'a ()>;
impl<C> EthernetWorker<C> {}
//~^ ERROR: the trait

fn main() {}
