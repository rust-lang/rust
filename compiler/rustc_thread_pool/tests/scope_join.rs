/// Test that one can emulate join with `scope`:
fn pseudo_join<F, G>(f: F, g: G)
where
    F: FnOnce() + Send,
    G: FnOnce() + Send,
{
    rayon_core::scope(|s| {
        s.spawn(|_| g());
        f();
    });
}

fn quick_sort<T: PartialOrd + Send>(v: &mut [T]) {
    if v.len() <= 1 {
        return;
    }

    let mid = partition(v);
    let (lo, hi) = v.split_at_mut(mid);
    pseudo_join(|| quick_sort(lo), || quick_sort(hi));
}

fn partition<T: PartialOrd + Send>(v: &mut [T]) -> usize {
    let pivot = v.len() - 1;
    let mut i = 0;
    for j in 0..pivot {
        if v[j] <= v[pivot] {
            v.swap(i, j);
            i += 1;
        }
    }
    v.swap(i, pivot);
    i
}

fn is_sorted<T: Send + Ord>(v: &[T]) -> bool {
    (1..v.len()).all(|i| v[i - 1] <= v[i])
}

#[test]
fn scope_join() {
    let mut v: Vec<i32> = (0..256).rev().collect();
    quick_sort(&mut v);
    assert!(is_sorted(&v));
}
