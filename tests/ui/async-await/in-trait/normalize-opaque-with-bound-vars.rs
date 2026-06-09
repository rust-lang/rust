//@ build-pass
//@ edition:2021
//@ compile-flags: -Cdebuginfo=2

// We were not normalizing opaques with escaping bound vars during codegen,
// leading to later errors during debuginfo computation.


#[derive(Clone, Copy)]
pub struct SharedState {}

pub trait State {
    #[allow(async_fn_in_trait)]
    async fn execute(self, shared_state: &SharedState);
}

pub trait StateComposer {
    fn and_then<T, F>(self, map_fn: F) -> AndThen<Self, F>
    where
        Self: State + Sized,
        T: State,
        F: FnOnce() -> T,
    {
        AndThen { previous: self, map_fn }
    }
}

impl<T> StateComposer for T where T: State {}
pub struct AndThen<T, F> {
    previous: T,
    map_fn: F,
}

impl<T, U, F> State for AndThen<T, F>
where
    T: State,
    U: State,
    F: FnOnce() -> U,
{
    async fn execute(self, shared_state: &SharedState)
    where
        Self: Sized,
    {
        self.previous.execute(shared_state).await;
        (self.map_fn)().execute(shared_state).await
    }
}

pub struct SomeState {}

impl State for SomeState {
    async fn execute(self, shared_state: &SharedState) {}
}

pub fn main() {
    let shared_state = SharedState {};
    async {
        SomeState {}
            .and_then(|| SomeState {})
            .and_then(|| SomeState {})
            .execute(&shared_state)
            .await;
    };
}
