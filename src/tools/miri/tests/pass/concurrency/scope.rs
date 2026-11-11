use std::thread;

fn main() {
    let mut a = vec![1, 2, 3];
    let mut x = 0;

    thread::scope(|s| {
        s.spawn(|| {
            // We can borrow `a` here.
            let _s = format!("hello from the first scoped thread: {a:?}");
        });
        s.spawn(|| {
            let _s = format!("hello from the second scoped thread");
            // We can even mutably borrow `x` here,
            // because no other threads are using it.
            x += a[0] + a[2];
        });
        let _s = format!("hello from the main thread");
    });

    // After the scope, we can modify and access our variables again:
    a.push(4);
    assert_eq!(x, a.len());
}
