fn my_fn(event: &Event<'_>) {}

struct Event<'a>(&'a ());

fn main() {
    const ptr: &fn(&Event<'_>) = &my_fn as _; //~ ERROR non-primitive cast: `&for<'a, 'b> fn(&'a Event<'b>) {my_fn}` as `&for<'a, 'b> fn(&'a Event<'b>)` [E0605]
}
