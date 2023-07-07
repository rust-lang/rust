// build-fail
// normalize-stderr-test "S32" -> "SXX"
// normalize-stderr-test "S1M" -> "SXX"
// error-pattern: too big for the current

struct S32<T> {
    v0: T,
    v1: T,
    v2: T,
    v3: T,
    v4: T,
    v5: T,
    v6: T,
    v7: T,
    v8: T,
    u9: T,
    v10: T,
    v11: T,
    v12: T,
    v13: T,
    v14: T,
    v15: T,
    v16: T,
    v17: T,
    v18: T,
    v19: T,
    v20: T,
    v21: T,
    v22: T,
    v23: T,
    v24: T,
    u25: T,
    v26: T,
    v27: T,
    v28: T,
    v29: T,
    v30: T,
    v31: T,
}

struct S1k<T> { val: S32<S32<T>> }

struct S1M<T> { val: S1k<S1k<T>> }

fn main() {
    let fat: Option<S1M<S1M<S1M<u32>>>> = None;
    //~^ ERROR are too big for the current architecture

}
