/**
 * Concurrency-enabled mechanisms for sharing mutable and/or immutable state
 * between tasks.
 */

import unsafe::{shared_mutable_state, clone_shared_mutable_state,
                get_shared_mutable_state, get_shared_immutable_state};

export arc, clone, get;

/****************************************************************************
 * Immutable ARC
 ****************************************************************************/

/// An atomically reference counted wrapper for shared immutable state.
struct arc<T: const send> { x: shared_mutable_state<T>; }

/// Create an atomically reference counted wrapper.
fn arc<T: const send>(+data: T) -> arc<T> {
    arc { x: unsafe { shared_mutable_state(data) } }
}

/**
 * Access the underlying data in an atomically reference counted
 * wrapper.
 */
fn get<T: const send>(rc: &arc<T>) -> &T {
    unsafe { get_shared_immutable_state(&rc.x) }
}

/**
 * Duplicate an atomically reference counted wrapper.
 *
 * The resulting two `arc` objects will point to the same underlying data
 * object. However, one of the `arc` objects can be sent to another task,
 * allowing them to share the underlying data.
 */
fn clone<T: const send>(rc: &arc<T>) -> arc<T> {
    arc { x: unsafe { clone_shared_mutable_state(&rc.x) } }
}

/****************************************************************************
 * Mutex protected ARC (unsafe)
 ****************************************************************************/

/****************************************************************************
 * R/W lock protected ARC
 ****************************************************************************/

/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    import comm::*;

    #[test]
    fn manually_share_arc() {
        let v = ~[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let arc_v = arc::arc(v);

        let p = port();
        let c = chan(p);

        do task::spawn() {
            let p = port();
            c.send(chan(p));

            let arc_v = p.recv();

            let v = *arc::get::<~[int]>(&arc_v);
            assert v[3] == 4;
        };

        let c = p.recv();
        c.send(arc::clone(&arc_v));

        assert (*arc::get(&arc_v))[2] == 3;

        log(info, arc_v);
    }
}
