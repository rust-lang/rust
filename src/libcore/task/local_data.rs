/*!

Task local data management

Allows storing boxes with arbitrary types inside, to be accessed anywhere
within a task, keyed by a pointer to a global finaliser function. Useful
for task-spawning metadata (tracking linked failure state), dynamic
variables, and interfacing with foreign code with bad callback interfaces.

To use, declare a monomorphic global function at the type to store, and use
it as the 'key' when accessing. See the 'tls' tests below for examples.

Casting 'Arcane Sight' reveals an overwhelming aura of Transmutation magic.

*/

export local_data_key;
export local_data_pop;
export local_data_get;
export local_data_set;
export local_data_modify;

// XXX: These shouldn't be exported but they are used by task.rs
export local_get;
export local_set;

/**
 * Indexes a task-local data slot. The function's code pointer is used for
 * comparison. Recommended use is to write an empty function for each desired
 * task-local data slot (and use class destructors, not code inside the
 * function, if specific teardown is needed). DO NOT use multiple
 * instantiations of a single polymorphic function to index data of different
 * types; arbitrary type coercion is possible this way.
 *
 * One other exception is that this global state can be used in a destructor
 * context to create a circular @-box reference, which will crash during task
 * failure (see issue #3039).
 *
 * These two cases aside, the interface is safe.
 */
type LocalDataKey<T: Owned> = &fn(+@T);

trait LocalData { }
impl<T: Owned> @T: LocalData { }

impl LocalData: Eq {
    pure fn eq(&&other: LocalData) -> bool unsafe {
        let ptr_a: (uint, uint) = cast::reinterpret_cast(&self);
        let ptr_b: (uint, uint) = cast::reinterpret_cast(&other);
        return ptr_a == ptr_b;
    }
    pure fn ne(&&other: LocalData) -> bool { !self.eq(other) }
}

// We use dvec because it's the best data structure in core. If TLS is used
// heavily in future, this could be made more efficient with a proper map.
type TaskLocalElement = (*libc::c_void, *libc::c_void, LocalData);
// Has to be a pointer at outermost layer; the foreign call returns void *.
type TaskLocalMap = @dvec::DVec<Option<TaskLocalElement>>;

extern fn cleanup_task_local_map(map_ptr: *libc::c_void) unsafe {
    assert !map_ptr.is_null();
    // Get and keep the single reference that was created at the beginning.
    let _map: TaskLocalMap = cast::reinterpret_cast(&map_ptr);
    // All local_data will be destroyed along with the map.
}

// Gets the map from the runtime. Lazily initialises if not done so already.
unsafe fn get_task_local_map(task: *rust_task) -> TaskLocalMap {

    // Relies on the runtime initialising the pointer to null.
    // NOTE: The map's box lives in TLS invisibly referenced once. Each time
    // we retrieve it for get/set, we make another reference, which get/set
    // drop when they finish. No "re-storing after modifying" is needed.
    let map_ptr = rustrt::rust_get_task_local_data(task);
    if map_ptr.is_null() {
        let map: TaskLocalMap = @dvec::DVec();
        // Use reinterpret_cast -- transmute would take map away from us also.
        rustrt::rust_set_task_local_data(
            task, cast::reinterpret_cast(&map));
        rustrt::rust_task_local_data_atexit(task, cleanup_task_local_map);
        // Also need to reference it an extra time to keep it for now.
        cast::bump_box_refcount(map);
        map
    } else {
        let map = cast::transmute(move map_ptr);
        cast::bump_box_refcount(map);
        map
    }
}

unsafe fn key_to_key_value<T: Owned>(
    key: LocalDataKey<T>) -> *libc::c_void {

    // Keys are closures, which are (fnptr,envptr) pairs. Use fnptr.
    // Use reintepret_cast -- transmute would leak (forget) the closure.
    let pair: (*libc::c_void, *libc::c_void) = cast::reinterpret_cast(&key);
    pair.first()
}

// If returning Some(..), returns with @T with the map's reference. Careful!
unsafe fn local_data_lookup<T: Owned>(
    map: TaskLocalMap, key: LocalDataKey<T>)
    -> Option<(uint, *libc::c_void)> {

    let key_value = key_to_key_value(key);
    let map_pos = (*map).position(|entry|
        match entry {
            Some((k,_,_)) => k == key_value,
            None => false
        }
    );
    do map_pos.map |index| {
        // .get() is guaranteed because of "None { false }" above.
        let (_, data_ptr, _) = (*map)[index].get();
        (index, data_ptr)
    }
}

unsafe fn local_get_helper<T: Owned>(
    task: *rust_task, key: LocalDataKey<T>,
    do_pop: bool) -> Option<@T> {

    let map = get_task_local_map(task);
    // Interpreturn our findings from the map
    do local_data_lookup(map, key).map |result| {
        // A reference count magically appears on 'data' out of thin air. It
        // was referenced in the local_data box, though, not here, so before
        // overwriting the local_data_box we need to give an extra reference.
        // We must also give an extra reference when not removing.
        let (index, data_ptr) = result;
        let data: @T = cast::transmute(move data_ptr);
        cast::bump_box_refcount(data);
        if do_pop {
            (*map).set_elt(index, None);
        }
        data
    }
}

unsafe fn local_pop<T: Owned>(
    task: *rust_task,
    key: LocalDataKey<T>) -> Option<@T> {

    local_get_helper(task, key, true)
}

unsafe fn local_get<T: Owned>(
    task: *rust_task,
    key: LocalDataKey<T>) -> Option<@T> {

    local_get_helper(task, key, false)
}

unsafe fn local_set<T: Owned>(
    task: *rust_task, key: LocalDataKey<T>, +data: @T) {

    let map = get_task_local_map(task);
    // Store key+data as *voids. Data is invisibly referenced once; key isn't.
    let keyval = key_to_key_value(key);
    // We keep the data in two forms: one as an unsafe pointer, so we can get
    // it back by casting; another in an existential box, so the reference we
    // own on it can be dropped when the box is destroyed. The unsafe pointer
    // does not have a reference associated with it, so it may become invalid
    // when the box is destroyed.
    let data_ptr = cast::reinterpret_cast(&data);
    let data_box = data as LocalData;
    // Construct new entry to store in the map.
    let new_entry = Some((keyval, data_ptr, data_box));
    // Find a place to put it.
    match local_data_lookup(map, key) {
        Some((index, _old_data_ptr)) => {
            // Key already had a value set, _old_data_ptr, whose reference
            // will get dropped when the local_data box is overwritten.
            (*map).set_elt(index, new_entry);
        }
        None => {
            // Find an empty slot. If not, grow the vector.
            match (*map).position(|x| x.is_none()) {
                Some(empty_index) => (*map).set_elt(empty_index, new_entry),
                None => (*map).push(new_entry)
            }
        }
    }
}

unsafe fn local_modify<T: Owned>(
    task: *rust_task, key: LocalDataKey<T>,
    modify_fn: fn(Option<@T>) -> Option<@T>) {

    // Could be more efficient by doing the lookup work, but this is easy.
    let newdata = modify_fn(local_pop(task, key));
    if newdata.is_some() {
        local_set(task, key, option::unwrap(move newdata));
    }
}

/* Exported interface for task-local data (plus local_data_key above). */
/**
 * Remove a task-local data value from the table, returning the
 * reference that was originally created to insert it.
 */
unsafe fn local_data_pop<T: Owned>(
    key: LocalDataKey<T>) -> Option<@T> {

    local_pop(rustrt::rust_get_task(), key)
}
/**
 * Retrieve a task-local data value. It will also be kept alive in the
 * table until explicitly removed.
 */
unsafe fn local_data_get<T: Owned>(
    key: LocalDataKey<T>) -> Option<@T> {

    local_get(rustrt::rust_get_task(), key)
}
/**
 * Store a value in task-local data. If this key already has a value,
 * that value is overwritten (and its destructor is run).
 */
unsafe fn local_data_set<T: Owned>(
    key: LocalDataKey<T>, +data: @T) {

    local_set(rustrt::rust_get_task(), key, data)
}
/**
 * Modify a task-local data value. If the function returns 'None', the
 * data is removed (and its reference dropped).
 */
unsafe fn local_data_modify<T: Owned>(
    key: LocalDataKey<T>,
    modify_fn: fn(Option<@T>) -> Option<@T>) {

    local_modify(rustrt::rust_get_task(), key, modify_fn)
}

#[test]
fn test_tls_multitask() unsafe {
    fn my_key(+_x: @~str) { }
    local_data_set(my_key, @~"parent data");
    do task::spawn unsafe {
        assert local_data_get(my_key).is_none(); // TLS shouldn't carry over.
        local_data_set(my_key, @~"child data");
        assert *(local_data_get(my_key).get()) == ~"child data";
        // should be cleaned up for us
    }
    // Must work multiple times
    assert *(local_data_get(my_key).get()) == ~"parent data";
    assert *(local_data_get(my_key).get()) == ~"parent data";
    assert *(local_data_get(my_key).get()) == ~"parent data";
}

#[test]
fn test_tls_overwrite() unsafe {
    fn my_key(+_x: @~str) { }
    local_data_set(my_key, @~"first data");
    local_data_set(my_key, @~"next data"); // Shouldn't leak.
    assert *(local_data_get(my_key).get()) == ~"next data";
}

#[test]
fn test_tls_pop() unsafe {
    fn my_key(+_x: @~str) { }
    local_data_set(my_key, @~"weasel");
    assert *(local_data_pop(my_key).get()) == ~"weasel";
    // Pop must remove the data from the map.
    assert local_data_pop(my_key).is_none();
}

#[test]
fn test_tls_modify() unsafe {
    fn my_key(+_x: @~str) { }
    local_data_modify(my_key, |data| {
        match data {
            Some(@val) => fail ~"unwelcome value: " + val,
            None       => Some(@~"first data")
        }
    });
    local_data_modify(my_key, |data| {
        match data {
            Some(@~"first data") => Some(@~"next data"),
            Some(@val)           => fail ~"wrong value: " + val,
            None                 => fail ~"missing value"
        }
    });
    assert *(local_data_pop(my_key).get()) == ~"next data";
}

#[test]
fn test_tls_crust_automorestack_memorial_bug() unsafe {
    // This might result in a stack-canary clobber if the runtime fails to set
    // sp_limit to 0 when calling the cleanup extern - it might automatically
    // jump over to the rust stack, which causes next_c_sp to get recorded as
    // Something within a rust stack segment. Then a subsequent upcall (esp.
    // for logging, think vsnprintf) would run on a stack smaller than 1 MB.
    fn my_key(+_x: @~str) { }
    do task::spawn {
        unsafe { local_data_set(my_key, @~"hax"); }
    }
}

#[test]
fn test_tls_multiple_types() unsafe {
    fn str_key(+_x: @~str) { }
    fn box_key(+_x: @@()) { }
    fn int_key(+_x: @int) { }
    do task::spawn unsafe {
        local_data_set(str_key, @~"string data");
        local_data_set(box_key, @@());
        local_data_set(int_key, @42);
    }
}

#[test]
fn test_tls_overwrite_multiple_types() {
    fn str_key(+_x: @~str) { }
    fn box_key(+_x: @@()) { }
    fn int_key(+_x: @int) { }
    do task::spawn unsafe {
        local_data_set(str_key, @~"string data");
        local_data_set(int_key, @42);
        // This could cause a segfault if overwriting-destruction is done with
        // the crazy polymorphic transmute rather than the provided finaliser.
        local_data_set(int_key, @31337);
    }
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_tls_cleanup_on_failure() unsafe {
    fn str_key(+_x: @~str) { }
    fn box_key(+_x: @@()) { }
    fn int_key(+_x: @int) { }
    local_data_set(str_key, @~"parent data");
    local_data_set(box_key, @@());
    do task::spawn unsafe { // spawn_linked
        local_data_set(str_key, @~"string data");
        local_data_set(box_key, @@());
        local_data_set(int_key, @42);
        fail;
    }
    // Not quite nondeterministic.
    local_data_set(int_key, @31337);
    fail;
}
