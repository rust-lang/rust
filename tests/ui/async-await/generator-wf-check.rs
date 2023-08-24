//! Tests if we draw implied bounds from the generator output or witness types,
//! and if so, whether we check these types are WF when checking the generator.

// edition: 2021
// revisions: output output_wf witness witness_wf
//[output] check-pass
//[output_wf] check-fail
//[witness] check-pass
//[witness_wf] check-fail

struct Static<T: 'static>(T);

fn wf<T>(_: T) {}

#[cfg(output)]
fn test_output<T>() {
    async {
        None::<Static<T>>
    };
}

#[cfg(output_wf)]
fn test_output_wf<T>() {
    wf(async {
    //[output_wf]~^ ERROR `T` may not live long enough
        None::<Static<T>>
    });
}

#[cfg(witness)]
fn test_witness<T>() {
    async {
        let witness: Option<Static<T>> = None;
        async {}.await;
        drop(witness);
    };
}

#[cfg(witness_wf)]
fn test_witness_wf<T>() {
    wf(async {
    //[witness_wf]~^ ERROR `T` may not live long enough
        let witness: Option<Static<T>> = None;
        async {}.await;
        drop(witness);
    });
}

fn main() {}
