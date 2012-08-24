/*!
 * Operations on the ubiquitous `option` type.
 *
 * Type `option` represents an optional value.
 *
 * Every `option<T>` value can either be `some(T)` or `none`. Where in other
 * languages you might use a nullable type, in Rust you would use an option
 * type.
 */

/// The option type
enum option<T> {
    none,
    some(T),
}

pure fn get<T: copy>(opt: option<T>) -> T {
    /*!
     * Gets the value out of an option
     *
     * # Failure
     *
     * Fails if the value equals `none`
     */

    match opt {
      some(x) => return x,
      none => fail ~"option::get none"
    }
}

pure fn get_ref<T>(opt: &r/option<T>) -> &r/T {
    /*!
     * Gets an immutable reference to the value inside an option.
     *
     * # Failure
     *
     * Fails if the value equals `none`
     */
    match *opt {
        some(ref x) => x,
        none => fail ~"option::get_ref none"
    }
}

pure fn expect<T: copy>(opt: option<T>, reason: ~str) -> T {
    /*!
     * Gets the value out of an option, printing a specified message on
     * failure
     *
     * # Failure
     *
     * Fails if the value equals `none`
     */
    match opt { some(x) => x, none => fail reason }
}

pure fn map<T, U>(opt: option<T>, f: fn(T) -> U) -> option<U> {
    //! Maps a `some` value from one type to another

    match opt { some(x) => some(f(x)), none => none }
}

pure fn map_ref<T, U>(opt: &option<T>, f: fn(x: &T) -> U) -> option<U> {
    //! Maps a `some` value by reference from one type to another

    match *opt { some(ref x) => some(f(x)), none => none }
}

pure fn map_consume<T, U>(+opt: option<T>, f: fn(+T) -> U) -> option<U> {
    /*!
     * As `map`, but consumes the option and gives `f` ownership to avoid
     * copying.
     */
    if opt.is_some() { some(f(option::unwrap(opt))) } else { none }
}

pure fn chain<T, U>(opt: option<T>, f: fn(T) -> option<U>) -> option<U> {
    /*!
     * Update an optional value by optionally running its content through a
     * function that returns an option.
     */

    match opt { some(x) => f(x), none => none }
}

pure fn chain_ref<T, U>(opt: &option<T>,
                        f: fn(x: &T) -> option<U>) -> option<U> {
    /*!
     * Update an optional value by optionally running its content by reference
     * through a function that returns an option.
     */

    match *opt { some(ref x) => f(x), none => none }
}

pure fn or<T>(+opta: option<T>, +optb: option<T>) -> option<T> {
    /*!
     * Returns the leftmost some() value, or none if both are none.
     */
    match opta {
        some(_) => opta,
        _ => optb
    }
}

#[inline(always)]
pure fn while_some<T>(+x: option<T>, blk: fn(+T) -> option<T>) {
    //! Applies a function zero or more times until the result is none.

    let mut opt <- x;
    while opt.is_some() {
        opt = blk(unwrap(opt));
    }
}

pure fn is_none<T>(opt: option<T>) -> bool {
    //! Returns true if the option equals `none`

    match opt { none => true, some(_) => false }
}

pure fn is_some<T>(opt: option<T>) -> bool {
    //! Returns true if the option contains some value

    !is_none(opt)
}

pure fn get_default<T: copy>(opt: option<T>, def: T) -> T {
    //! Returns the contained value or a default

    match opt { some(x) => x, none => def }
}

pure fn map_default<T, U>(opt: option<T>, +def: U, f: fn(T) -> U) -> U {
    //! Applies a function to the contained value or returns a default

    match opt { none => def, some(t) => f(t) }
}

// This should replace map_default.
pure fn map_default_ref<T, U>(opt: &option<T>, +def: U,
                              f: fn(x: &T) -> U) -> U {
    //! Applies a function to the contained value or returns a default

    match *opt { none => def, some(ref t) => f(t) }
}

// This should change to by-copy mode; use iter_ref below for by reference
pure fn iter<T>(opt: option<T>, f: fn(T)) {
    //! Performs an operation on the contained value or does nothing

    match opt { none => (), some(t) => f(t) }
}

pure fn iter_ref<T>(opt: &option<T>, f: fn(x: &T)) {
    //! Performs an operation on the contained value by reference
    match *opt { none => (), some(ref t) => f(t) }
}

#[inline(always)]
pure fn unwrap<T>(+opt: option<T>) -> T {
    /*!
     * Moves a value out of an option type and returns it.
     *
     * Useful primarily for getting strings, vectors and unique pointers out
     * of option types without copying them.
     */
    match move opt {
        some(move x) => x,
        none => fail ~"option::unwrap none"
    }
}

/// The ubiquitous option dance.
#[inline(always)]
fn swap_unwrap<T>(opt: &mut option<T>) -> T {
    if opt.is_none() { fail ~"option::swap_unwrap none" }
    unwrap(util::replace(opt, none))
}

pure fn unwrap_expect<T>(+opt: option<T>, reason: &str) -> T {
    //! As unwrap, but with a specified failure message.
    if opt.is_none() { fail reason.to_unique(); }
    unwrap(opt)
}

// Some of these should change to be &option<T>, some should not. See below.
impl<T> option<T> {
    /**
     * Update an optional value by optionally running its content through a
     * function that returns an option.
     */
    pure fn chain<U>(f: fn(T) -> option<U>) -> option<U> { chain(self, f) }
    /// Applies a function to the contained value or returns a default
    pure fn map_default<U>(+def: U, f: fn(T) -> U) -> U
        { map_default(self, def, f) }
    /// Performs an operation on the contained value or does nothing
    pure fn iter(f: fn(T)) { iter(self, f) }
    /// Returns true if the option equals `none`
    pure fn is_none() -> bool { is_none(self) }
    /// Returns true if the option contains some value
    pure fn is_some() -> bool { is_some(self) }
    /// Maps a `some` value from one type to another
    pure fn map<U>(f: fn(T) -> U) -> option<U> { map(self, f) }
}

impl<T> &option<T> {
    /**
     * Update an optional value by optionally running its content by reference
     * through a function that returns an option.
     */
    pure fn chain_ref<U>(f: fn(x: &T) -> option<U>) -> option<U> {
        chain_ref(self, f)
    }
    /// Applies a function to the contained value or returns a default
    pure fn map_default_ref<U>(+def: U, f: fn(x: &T) -> U) -> U
        { map_default_ref(self, def, f) }
    /// Performs an operation on the contained value by reference
    pure fn iter_ref(f: fn(x: &T)) { iter_ref(self, f) }
    /// Maps a `some` value from one type to another by reference
    pure fn map_ref<U>(f: fn(x: &T) -> U) -> option<U> { map_ref(self, f) }
    /// Gets an immutable reference to the value inside a `some`.
    pure fn get_ref() -> &self/T { get_ref(self) }
}

impl<T: copy> option<T> {
    /**
     * Gets the value out of an option
     *
     * # Failure
     *
     * Fails if the value equals `none`
     */
    pure fn get() -> T { get(self) }
    pure fn get_default(def: T) -> T { get_default(self, def) }
    /**
     * Gets the value out of an option, printing a specified message on
     * failure
     *
     * # Failure
     *
     * Fails if the value equals `none`
     */
    pure fn expect(reason: ~str) -> T { expect(self, reason) }
    /// Applies a function zero or more times until the result is none.
    pure fn while_some(blk: fn(+T) -> option<T>) { while_some(self, blk) }
}

#[test]
fn test_unwrap_ptr() {
    let x = ~0;
    let addr_x = ptr::addr_of(*x);
    let opt = some(x);
    let y = unwrap(opt);
    let addr_y = ptr::addr_of(*y);
    assert addr_x == addr_y;
}

#[test]
fn test_unwrap_str() {
    let x = ~"test";
    let addr_x = str::as_buf(x, |buf, _len| ptr::addr_of(buf));
    let opt = some(x);
    let y = unwrap(opt);
    let addr_y = str::as_buf(y, |buf, _len| ptr::addr_of(buf));
    assert addr_x == addr_y;
}

#[test]
fn test_unwrap_resource() {
    struct r {
       let i: @mut int;
       new(i: @mut int) { self.i = i; }
       drop { *(self.i) += 1; }
    }
    let i = @mut 0;
    {
        let x = r(i);
        let opt = some(x);
        let _y = unwrap(opt);
    }
    assert *i == 1;
}

#[test]
fn test_option_dance() {
    let x = some(());
    let mut y = some(5);
    let mut y2 = 0;
    do x.iter |_x| {
        y2 = swap_unwrap(&mut y);
    }
    assert y2 == 5;
    assert y.is_none();
}
#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_option_too_much_dance() {
    let mut y = some(util::NonCopyable());
    let _y2 = swap_unwrap(&mut y);
    let _y3 = swap_unwrap(&mut y);
}

#[test]
fn test_option_while_some() {
    let mut i = 0;
    do some(10).while_some |j| {
        i += 1;
        if (j > 0) {
            some(j-1)
        } else {
            none
        }
    }
    assert i == 11;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
