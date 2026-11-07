//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ compile-flags: -Zpolonius=off
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

struct InvokeOnDrop<T: Copy, F: FnMut(T)>(T, F);

impl<T: Copy, F: FnMut(T)> Drop for InvokeOnDrop<T, F> {
    fn drop(&mut self) {
        (self.1)(self.0);
    }
}

type Data = Vec<Vec<i32>>;
static mut LEAK: Option<&'static Data> = None;

fn main() {
    let long = Box::leak(Box::new(Data::new())) as &'static Data;
    {
        let local: Data = vec![vec![1]];
        let mut unified = [
            InvokeOnDrop(
                long,
                (|input: &'static Data| unsafe { LEAK = Some(input) }) as fn(&'static Data),
            ),
            InvokeOnDrop(&local, (|_: &Data| {}) as fn(&Data)),
            //~^ ERROR `local` does not live long enough
        ];
        let [left, right] = &mut unified;
        std::mem::swap(&mut left.0, &mut right.0);
    }
    let escaped = unsafe { LEAK.unwrap() };
    println!("{escaped:?}");
}
