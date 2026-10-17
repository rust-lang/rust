/*! ```compile_fail,E0524

fn quick_sort<T:PartialOrd+Send>(v: &mut [T]) {
    if v.len() <= 1 {
        return;
    }

    let mid = partition(v);
    let (_lo, hi) = v.split_at_mut(mid);
    rustc_thread_pool::join(|| quick_sort(hi), || quick_sort(hi)); //~ ERROR
}

fn partition<T:PartialOrd+Send>(v: &mut [T]) -> usize {
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

fn main() { }

``` */
