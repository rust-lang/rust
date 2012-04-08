export anymap, seqmap, parmap;

fn anymap<T:send, U:send>(v: [T], f: fn~(T) -> U) -> [U] {
    parmap(v, f)
}

fn seqmap<T, U>(v: [T], f: fn(T) -> U) -> [U] {
    vec::map(v, f)
}

fn parmap<T:send, U:send>(v: [T], f: fn~(T) -> U) -> [U] unsafe {
    let futures = vec::map(v) {|elt|
        let po = comm::port();
        let ch = comm::chan(po);
        let addr = ptr::addr_of(elt);
        task::spawn {||
            comm::send(ch, f(*addr));
        }
        po
    };
    vec::map(futures) {|future|
        comm::recv(future)
    }
}

#[test]
fn test_parallel_map() {
    let i = [1, 2, 3, 4];
    let j = parmap(i) {|e| e + 1 };
    assert j == [2, 3, 4, 5];
}