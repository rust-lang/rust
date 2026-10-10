//@ known-bug: #124340
#![feature(anonymous_lifetime_in_impl_trait)]

trait Producer {
    type Output;
    fn produce(self) -> Self::Output;
}

trait SomeTrait<'a> {}

fn force_same_lifetime<'a>(_x: &'a i32, _y: impl SomeTrait<'a>) {
    unimplemented!()
}

fn foo<'a>(s: &'a i32, producer: impl Producer<Output: SomeTrait<'_>>) {
    force_same_lifetime(s, producer.produce());
}
