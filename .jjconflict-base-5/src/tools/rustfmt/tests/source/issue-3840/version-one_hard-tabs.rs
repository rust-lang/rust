// rustfmt-hard_tabs: true

impl<Target: FromEvent<A> + FromEvent<B>, A: Widget2<Ctx = C>, B: Widget2<Ctx = C>, C: for<'a> CtxFamily<'a>> Widget2 for WidgetEventLifter<Target, A, B>
{
    type Ctx = C;
    type Event = Vec<Target>;
}

mod foo {
    impl<Target: FromEvent<A> + FromEvent<B>, A: Widget2<Ctx = C>, B: Widget2<Ctx = C>, C: for<'a> CtxFamily<'a>> Widget2 for WidgetEventLifter<Target, A, B>
    {
        type Ctx = C;
        type Event = Vec<Target>;
    }
}
