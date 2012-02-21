export parmap;

fn parmap<T:send, U:send>(v: [T], f: fn~(T) -> U) -> [U] {
    let futures = vec::map(v) {|elt|
        future::spawn {||
            f(elt)
        }
    };
    vec::map(futures) {|future|
        future::get(future)
    }
}

#[test]
fn test_parallel_map() {
    let i = [1, 2, 3, 4];
    let j = parmap(i) {|e| e + 1 };
    assert j == [2, 3, 4, 5];
}