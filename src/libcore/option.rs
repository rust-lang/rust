#[doc = "
Operations on the ubiquitous `option` type.

Type `option` represents an optional value.

Every `option<T>` value can either be `some(T)` or `none`. Where in other
languages you might use a nullable type, in Rust you would use an option type.
"];

#[doc = "The option type"]
enum option<T> {
    none,
    some(T),
}

pure fn get<T: copy>(opt: option<T>) -> T {
    #[doc = "
    Gets the value out of an option

    # Failure

    Fails if the value equals `none`
    "];

    alt opt { some(x) { ret x; } none { fail "option none"; } }
}

fn map<T, U: copy>(opt: option<T>, f: fn(T) -> U) -> option<U> {
    #[doc = "Maps a `some` value from one type to another"];

    alt opt { some(x) { some(f(x)) } none { none } }
}

fn chain<T, U>(opt: option<T>, f: fn(T) -> option<U>) -> option<U> {
    #[doc = "
    Update an optional value by optionally running its content through a
    function that returns an option.
    "];

    alt opt { some(x) { f(x) } none { none } }
}

pure fn is_none<T>(opt: option<T>) -> bool {
    #[doc = "Returns true if the option equals `none`"];

    alt opt { none { true } some(_) { false } }
}

pure fn is_some<T>(opt: option<T>) -> bool {
    #[doc = "Returns true if the option contains some value"];

    !is_none(opt)
}

pure fn get_default<T: copy>(opt: option<T>, def: T) -> T {
    #[doc = "Returns the contained value or a default"];

    alt opt { some(x) { x } none { def } }
}

fn map_default<T, U: copy>(opt: option<T>, def: U, f: fn(T) -> U) -> U {
    #[doc = "Applies a function to the contained value or returns a default"];

    alt opt { none { def } some(t) { f(t) } }
}

fn iter<T>(opt: option<T>, f: fn(T)) {
    #[doc = "Performs an operation on the contained value or does nothing"];

    alt opt { none { } some(t) { f(t); } }
}

fn unwrap<T>(-opt: option<T>) -> T unsafe {
    #[doc = "
    Moves a value out of an option type and returns it.

    Useful primarily for getting strings, vectors and unique pointers out of
    option types without copying them.
    "];

    let addr = alt opt {
      some(x) { ptr::addr_of(x) }
      none { fail "option none" }
    };
    let liberated_value = unsafe::reinterpret_cast(*addr);
    unsafe::forget(opt);
    ret liberated_value;
}

impl extensions<T:copy> for option<T> {
    #[doc = "
    Update an optional value by optionally running its content through a
    function that returns an option.
    "]
    fn chain<U>(f: fn(T) -> option<U>) -> option<U> { chain(self, f) }
    #[doc = "Returns the contained value or a default"]
    fn get_default(def: T) -> T { get_default(self, def) }
    #[doc = "Applies a function to the contained value or returns a default"]
    fn map_default<U: copy>(def: U, f: fn(T) -> U) -> U
        { map_default(self, def, f) }
    #[doc = "Performs an operation on the contained value or does nothing"]
    fn iter(f: fn(T)) { iter(self, f) }

    #[doc = "Converts `none` to a zero-element list and `some` to a \
             one-element list."]
    fn to_vec() -> [T] {
        alt self {
          some(t) { [t] }
          none { [] }
        }
    }

    #[doc = "Performs an operation on the contained value or does nothing"]
    fn each(f: fn(T) -> bool) {
        alt self {
          none { /* ok */ }
          some(e) { f(e); }
        }
    }

    #[doc = "
    Gets the value out of an option

    # Failure

    Fails if the value equals `none`
    "]
    fn get() -> T { get(self) }
    #[doc = "Returns true if the option equals `none`"]
    fn is_none() -> bool { is_none(self) }
    #[doc = "Returns true if the option contains some value"]
    fn is_some() -> bool { is_some(self) }
    #[doc = "Maps a `some` value from one type to another"]
    fn map<U:copy>(f: fn(T) -> U) -> option<U> { map(self, f) }
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
    let x = "test";
    let addr_x = str::as_buf(x) {|buf| ptr::addr_of(buf) };
    let opt = some(x);
    let y = unwrap(opt);
    let addr_y = str::as_buf(y) {|buf| ptr::addr_of(buf) };
    assert addr_x == addr_y;
}

#[test]
fn test_unwrap_resource() {
    resource r(i: @mut int) {
        *i += 1;
    }
    let i = @mut 0;
    {
        let x = r(i);
        let opt = some(x);
        let _y = unwrap(opt);
    }
    assert *i == 1;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
