// check-pass

trait Yokeable<'a> {
    type Output: 'a;
}
impl<'a> Yokeable<'a> for () {
    type Output = ();
}

trait DataMarker<'data> {
    type Yokeable: for<'a> Yokeable<'a>;
}
impl<'data> DataMarker<'data> for () {
    type Yokeable = ();
}

struct DataPayload<'data, M>(&'data M);

impl DataPayload<'static, ()> {
    pub fn map_project_with_capture<M2, T>(
        _: for<'a> fn(
            capture: T,
            std::marker::PhantomData<&'a ()>,
        ) -> <M2::Yokeable as Yokeable<'a>>::Output,
    ) -> DataPayload<'static, M2>
    where
        M2: DataMarker<'static>,
    {
        todo!()
    }
}

fn main() {
    let _: DataPayload<()> = DataPayload::<()>::map_project_with_capture::<_, &()>(|_, _| todo!());
}
