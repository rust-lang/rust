/*!
Global data

An interface for creating and retrieving values with global
(per-runtime) scope.

Global values are stored in a map and protected by a single global
mutex. Operations are provided for accessing and cloning the value
under the mutex.

Because all globals go through a single mutex, they should be used
sparingly.  The interface is intended to be used with clonable,
atomically reference counted synchronization types, like ARCs, in
which case the value should be cached locally whenever possible to
avoid hitting the mutex.
*/

use cast::{transmute, reinterpret_cast};
use clone::Clone;
use kinds::Owned;
use libc::{c_void, uintptr_t};
use option::{Option, Some, None};
use ops::Drop;
use pipes;
use private::{Exclusive, exclusive};
use private::{SharedMutableState, shared_mutable_state};
use private::{get_shared_immutable_state};
use private::at_exit::at_exit;
use send_map::linear::LinearMap;
use sys::Closure;
use task::spawn;
use uint;

pub type GlobalDataKey<T: Owned> = &fn(v: T);

pub unsafe fn global_data_clone_create<T: Owned Clone>(
    key: GlobalDataKey<T>, create: &fn() -> ~T) -> T {
    /*!
     * Clone a global value or, if it has not been created,
     * first construct the value then return a clone.
     *
     * # Safety note
     *
     * Both the clone operation and the constructor are
     * called while the global lock is held. Recursive
     * use of the global interface in either of these
     * operations will result in deadlock.
     */
    global_data_clone_create_(key_ptr(key), create)
}

unsafe fn global_data_clone_create_<T: Owned Clone>(
    key: uint, create: &fn() -> ~T) -> T {

    let mut clone_value: Option<T> = None;
    do global_data_modify_(key) |value: Option<~T>| {
        match value {
            None => {
                let value = create();
                clone_value = Some(value.clone());
                Some(value)
            }
            Some(value) => {
                clone_value = Some(value.clone());
                Some(value)
            }
        }
    }
    return clone_value.unwrap();
}

unsafe fn global_data_modify<T: Owned>(
    key: GlobalDataKey<T>, op: &fn(Option<~T>) -> Option<~T>) {

    global_data_modify_(key_ptr(key), op)
}

unsafe fn global_data_modify_<T: Owned>(
    key: uint, op: &fn(Option<~T>) -> Option<~T>) {

    let mut old_dtor = None;
    do get_global_state().with |gs| unsafe {
        let (maybe_new_value, maybe_dtor) = match gs.map.pop(&key) {
            Some((ptr, dtor)) => {
                let value: ~T = transmute(ptr);
                (op(Some(value)), Some(dtor))
            }
            None => {
                (op(None), None)
            }
        };
        match maybe_new_value {
            Some(value) => {
                let data: *c_void = transmute(value);
                let dtor: ~fn() = match maybe_dtor {
                    Some(dtor) => dtor,
                    None => {
                        let dtor: ~fn() = || unsafe {
                            let _destroy_value: ~T = transmute(data);
                        };
                        dtor
                    }
                };
                let value = (data, dtor);
                gs.map.insert(key, value);
            }
            None => {
                match maybe_dtor {
                    Some(dtor) => old_dtor = Some(dtor),
                    None => ()
                }
            }
        }
    }
}

pub unsafe fn global_data_clone<T: Owned Clone>(
    key: GlobalDataKey<T>) -> Option<T> {
    let mut maybe_clone: Option<T> = None;
    do global_data_modify(key) |current| {
        match &current {
            &Some(~ref value) => {
                maybe_clone = Some(value.clone());
            }
            &None => ()
        }
        current
    }
    return maybe_clone;
}

// GlobalState is a map from keys to unique pointers and a
// destructor. Keys are pointers derived from the type of the
// global value.  There is a single GlobalState instance per runtime.
struct GlobalState {
    map: LinearMap<uint, (*c_void, ~fn())>
}

impl GlobalState: Drop {
    fn finalize(&self) {
        for self.map.each_value |v| {
            match v {
                &(_, ref dtor) => (*dtor)()
            }
        }
    }
}

fn get_global_state() -> Exclusive<GlobalState> unsafe {

    const POISON: int = -1;

    // XXX: Doing atomic_cxchg to initialize the global state
    // lazily, which wouldn't be necessary with a runtime written
    // in Rust
    let global_ptr = rust_get_global_data_ptr();

    if *global_ptr == 0 {
        // Global state doesn't exist yet, probably

        // The global state object
        let state = GlobalState {
            map: LinearMap()
        };

        // It's under a reference-counted mutex
        let state = ~exclusive(state);

        // Convert it to an integer
        let state_ptr: &Exclusive<GlobalState> = state;
        let state_i: int = transmute(state_ptr);

        // Swap our structure into the global pointer
        let prev_i = atomic_cxchg(&mut *global_ptr, 0, state_i);

        // Sanity check that we're not trying to reinitialize after shutdown
        assert prev_i != POISON;

        if prev_i == 0 {
            // Successfully installed the global pointer

            // Take a handle to return
            let clone = state.clone();

            // Install a runtime exit function to destroy the global object
            do at_exit || unsafe {
                // Poison the global pointer
                let prev_i = atomic_cxchg(&mut *global_ptr, state_i, POISON);
                assert prev_i == state_i;

                // Capture the global state object in the at_exit closure
                // so that it is destroyed at the right time
                let _capture_global_state = &state;
            };
            return clone;
        } else {
            // Somebody else initialized the globals first
            let state: &Exclusive<GlobalState> = transmute(prev_i);
            return state.clone();
        }
    } else {
        let state: &Exclusive<GlobalState> = transmute(*global_ptr);
        return state.clone();
    }
}

fn key_ptr<T: Owned>(key: GlobalDataKey<T>) -> uint unsafe {
    let closure: Closure = reinterpret_cast(&key);
    return transmute(closure.code);
}

extern {
    fn rust_get_global_data_ptr() -> *mut int;
}

#[abi = "rust-intrinsic"]
extern {
    fn atomic_cxchg(dst: &mut int, old: int, src: int) -> int;
}

#[test]
fn test_clone_rc() unsafe {
    type MyType = SharedMutableState<int>;

    fn key(_v: SharedMutableState<int>) { }

    for uint::range(0, 100) |_| {
        do spawn unsafe {
            let val = do global_data_clone_create(key) {
                ~shared_mutable_state(10)
            };

            assert get_shared_immutable_state(&val) == &10;
        }
    }
}

#[test]
fn test_modify() unsafe {
    type MyType = SharedMutableState<int>;

    fn key(_v: SharedMutableState<int>) { }

    do global_data_modify(key) |v| unsafe {
        match v {
            None => {
                Some(~shared_mutable_state(10))
            }
            _ => fail
        }
    }

    do global_data_modify(key) |v| {
        match v {
            Some(sms) => {
                let v = get_shared_immutable_state(sms);
                assert *v == 10;
                None
            },
            _ => fail
        }
    }

    do global_data_modify(key) |v| unsafe {
        match v {
            None => {
                Some(~shared_mutable_state(10))
            }
            _ => fail
        }
    }
}
