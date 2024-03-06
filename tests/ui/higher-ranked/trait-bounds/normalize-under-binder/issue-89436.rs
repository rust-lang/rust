//@ check-pass

#![allow(unused)]

trait MiniYokeable<'a> {
    type Output;
}

struct MiniYoke<Y: for<'a> MiniYokeable<'a>> {
    pub yokeable: Y,
}

fn map_project_broken<Y, P>(
    source: MiniYoke<Y>,
    f: impl for<'a> FnOnce(
        <Y as MiniYokeable<'a>>::Output,
        core::marker::PhantomData<&'a ()>,
    ) -> <P as MiniYokeable<'a>>::Output,
) -> MiniYoke<P>
where
    Y: for<'a> MiniYokeable<'a>,
    P: for<'a> MiniYokeable<'a>
{
    unimplemented!()
}

struct Bar<'a> {
    string_1: &'a str,
    string_2: &'a str,
}

impl<'a> MiniYokeable<'a> for Bar<'static> {
    type Output = Bar<'a>;
}

impl<'a> MiniYokeable<'a> for &'static str {
    type Output = &'a str;
}

fn demo_broken(bar: MiniYoke<Bar<'static>>) -> MiniYoke<&'static str> {
    map_project_broken(bar, |bar, _| bar.string_1)
}

fn main() {}
