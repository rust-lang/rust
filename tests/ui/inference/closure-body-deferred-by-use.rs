//! Closure body checking should follow inference state rather than the closure's syntax position.
//! A direct call can constrain unresolved inputs before the body is checked.

//@ check-pass

fn take_getter<F: FnOnce(Value) -> i32>(_: F) {}

#[derive(Copy, Clone)]
struct Value;

impl Value {
    fn get(self) -> i32 {
        0
    }
}

struct Converted(u64);

impl Converted {
    fn from_value(value: u64) -> Option<Self> {
        Some(Self(value))
    }

    fn value(&self) -> u64 {
        self.0
    }
}

fn make<T>() -> Option<T> {
    None
}

fn closure_consumed_by_adapter() -> Result<Converted, ()> {
    let converted = make()
        .and_then(|value| Converted::from_value(value))
        .ok_or(())?;
    let _ = converted.value();
    Ok(converted)
}

fn main() {
    let array = [1_i64];

    let _: usize = (|range| range.count())(0_u8..5);

    let from_block = { |index| array[index].pow(1) };
    let _: i64 = from_block(0_usize);

    let tupled = (|index| array[index].pow(1),);
    let _: i64 = tupled.0(0_usize);

    let assigned;
    assigned = |index| array[index].pow(1);
    let _: i64 = assigned(0_usize);

    // Inline closure arguments still receive the call site's existing expectations.
    take_getter(|value| value.get());

    let _ = closure_consumed_by_adapter();
}
