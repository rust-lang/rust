//! Test basic newtype pattern functionality.

//@ run-pass

#[derive(Copy, Clone)]
struct Counter(CounterData);

#[derive(Copy, Clone)]
struct CounterData {
    compute: fn(Counter) -> isize,
    val: isize,
}

fn compute_value(counter: Counter) -> isize {
    let Counter(data) = counter;
    data.val + 20
}

pub fn main() {
    let my_counter = Counter(CounterData { compute: compute_value, val: 30 });

    // Test destructuring and function pointer call
    let Counter(data) = my_counter;
    assert_eq!((data.compute)(my_counter), 50);
}
